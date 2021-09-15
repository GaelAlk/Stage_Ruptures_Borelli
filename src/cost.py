import itertools
import random

import numpy as np
import ruptures as rpt
from cvxopt import matrix, solvers


from scipy.optimize import minimize
from ruptures.utils import sanity_check

class Mahalanobis(rpt.base.BaseCost):
    """
    Mahalanobis cost function
    """

    model = "Mahalanobis"
    min_size = 3


    def __init__(self,M):
        self.M=M
        if np.linalg.matrix_rank(self.M)==self.M.shape[0]:
            self.rang_plein=True
            self.L=np.linalg.cholesky(M)
        else :
            self.rang_plein=False
    def fit(self, signal):
        """Compute params to segment signal.
        Args:
            signal (array): signal to segment.
        """
        if self.rang_plein:
            self.signal = signal@self.L
        else:
            self.signal = signal
    def error(self, start, end):
        """Return the approximation cost on the segment [start:end].

        Args:
            start (int): start of the segment
            end (int): end of the segment

        Returns:
            segment cost
        """
        subsignal = self.signal[start:end]

        if self.rang_plein:
            return subsignal.var(axis=0).sum() * (end-start)
        else:

            submean=np.mean(subsignal,axis=0)
            matrice=(subsignal@self.M).dot(subsignal.T)
            return np.trace(matrice)-(np.sum(matrice))/(end-start)

class SML(Mahalanobis):
    def __init__(self,signals,bkps,nb_annotation=2000,mu=1000,gamma=10,eps=1,maxit=50):
        n=len(signals)
        M=self.SML_Matrix(signals,bkps,0.5*np.ones(n*int(nb_annotation/n)),mu,eps,gamma,maxit,int(nb_annotation/n))
        super().__init__(M)
    def X(self,signal,tau):
        xi=signal[tau[0]].reshape(-1,1)
        xj=signal[tau[1]].reshape(-1,1)
        xk=signal[tau[2]].reshape(-1,1)
        xij=xi-xj
        xjk=xj-xk
        Xt=xjk@xjk.T-xij@xij.T
        return Xt
    def CalculL(self,signal,mu,listtau):
        somme=0
        listxt=[]
        for tau in listtau:
            Xt=self.X(signal,tau)
            listxt.append(Xt)
            somme=somme+np.linalg.norm(Xt,ord="fro")**2
        return somme/(2*mu),listxt

    def CalculM(self,lambds,U,gamma,mu,T):
        n=lambds.shape[0]
        mat= mu*np.eye(n)+gamma*(np.tril(np.ones((n,n)))-np.eye(n))
        contr=np.sqrt(T/gamma)
        Q = matrix(2*mat)
        p = matrix(-lambds)
        G = matrix(np.vstack((np.ones((n,n)),-np.eye(n))))
        h = matrix(np.hstack((contr*np.ones(n),np.zeros(n))))
        sol=solvers.qp(Q, p, G, h,options={'show_progress': False},kktsolver="chol")
        solution=[np.array(sol['x'])[i][0] for i in range(n)]

        return U@np.diag(solution)@U.T

    def CalculZ(self,gradient,u,L):
        n=u.shape[0]
        Q = matrix(L*np.eye(n,n))
        p = matrix(gradient-1*u)
        G = matrix(np.vstack((np.eye(n),-np.eye(n))))
        h = matrix(np.hstack((np.ones(n),np.zeros(n))))
        sol=solvers.qp(Q, p, G, h,options={'show_progress': False})
        solution=[np.array(sol['x'])[i][0] for i in range(n)]

        return np.array(solution)

    def CalculV(self,listgradient,listu,L):
        n=listu[0].shape[0]
        Q = matrix(L*np.eye(n,n))
        vecteur=np.sum([0.5*(i+1)*listgradient[i] for i in range(len(listgradient))],0)
        vecteur=np.sum([-listu[0],vecteur],0)
        p = matrix(vecteur)
        G = matrix(np.vstack((np.eye(n),-np.eye(n))))
        h = matrix(np.hstack((np.ones(n),np.zeros(n))))
        sol=solvers.qp(Q, p, G, h,options={'show_progress': False})
        solution=[np.array(sol['x'])[i][0] for i in range(n)]


        return np.array(solution)

    def frobinius_scalar_product(self,A,B):
        """
        Frobinius scalar product between A and B
        """
        return np.trace(A.T@B)

    def CalculGradient(self,Mmu,listxt):
        grad=[]
        for Xt in listxt:
            grad.append(-1+self.frobinius_scalar_product(Xt,Mmu))
        return np.array(grad)
    def tripletsindice(self,signal,bkps,anot=0.1):
        labelsITML,anotITML=self.create_labels([bkps],anot)
        signalo=[]
        simpairs=[]
        dispairs=[]

        for i,ssvect in enumerate(anotITML[0]):

            signalo.append(signal[0][ssvect[0]:ssvect[1]])
            if(i>0):
                sim=list(itertools.combinations(np.arange(ssvect[0],ssvect[1]),2))
                dispairs.append(list(itertools.product(sim,ancien)))

            ancien=np.arange(ssvect[0],ssvect[1])

        dispairs=np.vstack(list(dispairs))
        dispairs=[np.hstack((dispairs[i][0],dispairs[i][1])) for i in range(dispairs.shape[0])]
        return dispairs
    def create_labels(self,bkps_list, annotation_ratio=0.1):
        """Create annotations centered inside each breaking points. Non annotated part have a -1 label

        Args:
            bkps_list (list): list of indexes of breaking points
            annotation_ratio (float, optional): ratio of labelled part. Defaults to 1.0.

        Returns:
            list: labels
        """
        UNLABELLED_IDX = -1

        labels_list = []
        annotations_list=[]
        for bkps in bkps_list:
            bkps = [0] + sorted(bkps)
            labels = np.full((bkps[-1], 1), UNLABELLED_IDX)
            annotation=[]
            for idx, (start, end) in enumerate(rpt.utils.pairwise(bkps)):

                offset = int((end - start) * (1 - annotation_ratio) // 2)
                labels[start + offset : end - offset] = idx


                annotation.append([start+offset,end-offset])
            annotations_list.append(annotation)
            labels_list.append(labels)
        return labels_list,annotations_list
    def SML_Matrix(self,signal,bkps,u0,mu,eps,gamma,maxit,nb=500,anoted=0.1):
        M=np.zeros((signal[0].shape[1],signal[0].shape[1]))
        t=0
        pairs=[]
        listtau=[]
        listxt=[]
        L=0

        for j,sign in enumerate(signal):
            pairstemp=self.tripletsindice([sign],bkps[j],anot=anoted)
            for i in pairstemp:
                pairs.append(i)
            listtautemp=np.array(pairstemp)[np.random.choice(np.arange(len(pairstemp)),nb)]
            for i in listtautemp:
                listtau.append(i)

            Ltemp,listxttemp=self.CalculL(sign,mu,listtautemp)
            L=L+Ltemp
            for i in listxttemp:
                listxt.append(i)
        L=L/(len(signal))
        pairs=np.array(pairs)
        listtau=np.array(listtau)
        listxt=np.array(listxt)
        T=listtau.shape[0]
        listu=[u0]
        listgradient=[]
        u=u0

        for t in range(maxit):
            matriceeigh=np.sum([i*j for (i,j) in zip(u,listxt)],0)
            lambds,U=np.linalg.eigh(matriceeigh)
            Mmu=self.CalculM(lambds,U,gamma,mu,T)
            gradient=self.CalculGradient(Mmu,listxt)
            listgradient.append(gradient)
            oldM=M
            M=M*(t/(t+2))+Mmu*(2/(t+2))
            z=self.CalculZ(gradient,u,L)
            v=self.CalculV(listgradient,listu,L)
            u=(2/(t+3))*v+((t+1)/(t+3))*z
            listu.append(u)
            if np.linalg.norm(M-oldM)<eps:
                break
        return M


class ITML_Kernel(rpt.base.BaseCost):
    """"""

    model = "Learned Kernel Partial Annotation"
    min_size = 3

    def pre_fit(
        self,
        signals,
        labels,
        initial_kernel_fct,
        upper_bound_similarity,
        lower_bound_dissimilarity,
        gamma,
    ):
        """computes the parameters (G_hat, G, training_samples) of the learned metrics

        Args:
            signals (List[array]): signals on which the metric is learned. List of len n_signals of
                array of shape(n_samples, n_features)
            labels (List[array]): corresponding labels of the signals. List of len n_signals of
                array of shape (n_samples, 1). The labels must be integers (>=0). The non-labelled
                samples can be anything below 0.
            upper_bound_similarity: [Bregman param] upper bound for the similarity constrains
            lower_bound_dissimilarity: [Bregman param] lower bound for the dissimilarity constrains
            gamma: [Bregman param] tradeoff between satisfying the constraints and minimizing DKL(G_hat,G)
        """
        self.initial_kernel_fct = initial_kernel_fct
        self.u = upper_bound_similarity
        self.l = lower_bound_dissimilarity
        self.gamma = gamma

        self.training_samples, self.constrains = self.get_training_samples_and_constains(
            signals, labels
        )

        self.G = initial_kernel_fct(self.training_samples, self.training_samples)

        self.G_inv = np.linalg.pinv(self.G)

        self.G_hat = self.MetricWithPartialAnnotationCostcompute_bregman()

        self.G_core = self.G_inv @ (self.G_hat - self.G) @ self.G_inv

    def fit(self, signal):
        """Compute params to segment signal.
        Args:
            signal (array): signal to segment.
        """
        self.signal = signal

    def error(self, start, end):
        """Return the approximation cost on the segment [start:end].

        Args:
            start (int): start of the segment
            end (int): end of the segment

        Returns:
            segment cost
        """
        # compute equation 8.7) and then 8.8) in Charles Truong. Détection de ruptures multiples –
        # application aux signaux physiologiques.

        subsignal = self.signal[start:end]

        self.inner_product = self.initial_kernel_fct(subsignal, subsignal)

        self.inner_product_with_training_samples = self.initial_kernel_fct(
            subsignal, self.training_samples
        )

        # TODO: optimisation replace np.diag(self.initial_kernel_fct(subsignal, subsignal)) by
        # self.initial_kernel_fct.diag(subsignal)
        inner_product_sum = np.sum(np.diag(self.inner_product)) - 1.0 / (end - start) * np.sum(
            self.inner_product
        )
        second_term = (
            self.inner_product_with_training_samples
            @ self.G_core
            @ self.inner_product_with_training_samples.T
        )
        new_kernel_product = np.sum(np.diag(second_term)) - 1.0 / (end - start) * np.sum(
            second_term
        )
        cost_bis = inner_product_sum + new_kernel_product

        return cost_bis

    def _phi_m_hat_phi(self, i, j):
        ki = self.inner_product_with_training_samples[i, :][np.newaxis, :]
        kj = self.inner_product_with_training_samples[j, :][np.newaxis, :]

        return self.inner_product[i, j] + (ki @ self.G_core @ kj.T)[0][0]

    @staticmethod
    def get_training_samples_and_constains(signals, labels):
        """Derives training samples and constrains dictionnary from labels list.

        Args:
            signals (List[array]): signals on which the metric is learned. List of len n_signals of
                array of shape(n_samples, n_features)
            labels (List[array]): corresponding labels of the signals. List of len n_signals of
                array of shape (n_samples, 1). The labels must be integers (>=0).The non-labelled
                samples can be anything below 0.

        Returns:
            (list, dict): list containing the training samples and a dictionnary whose keys
                corresponds to indexes in the list and the value to the label 1 (similar) or -1
                (dissimilar)
        """
        training_samples = []
        constrains = {}
        last_idx = -1
        for signal, label in zip(signals, labels):
            begin_signal = True
            idx_max = np.max(label)
            for idx in range(idx_max + 1):

                sub_signal = signal[(label == idx).squeeze()]
                new_idx_iterator = range(last_idx + 1, sub_signal.shape[0] + last_idx + 1)

                for key in itertools.combinations(new_idx_iterator, r=2):
                    constrains[key] = 1  # similar

                if begin_signal:
                    begin_signal = False
                else:
                    for key in itertools.product(last_idx_iterator, new_idx_iterator):
                        constrains[key] = -1  # disimilar

                training_samples.append(sub_signal)
                last_idx_iterator = new_idx_iterator
                *_, last_idx = last_idx_iterator

        training_samples = np.concatenate(training_samples)
        print(f"training_samples.shape: {training_samples.shape}")

        return training_samples, constrains

    def compute_bregman(self):
        # Algorithm 1 in P. Jain, B. Kulis, J. V. Davis, and I. S. Dhillon, “Metric and kernel
        # learning using a linear transformation,” Journal of Machine Learning Research (JMLR), vol.
        #  13, pp.519–547, 2012.

        eps = np.finfo(float).eps

        K = self.G.copy()
        lambdas = dict.fromkeys(self.constrains.keys(), np.array(0))
        xi = dict(
            [
                (key, self.u) if value == 1 else (key, self.l)
                for key, value in self.constrains.items()
            ]
        )

        n = K.shape[0]
        convergence = True

        min_ite = int(len(self.constrains)/16)
        num_ite = 0

        while convergence:
            num_ite += 1
            print(f"Iteration numéro°: {num_ite}")
            convergence = False
            upd=[]
            for _ in range(min_ite):

                (i, j), delta = random.choice(list(self.constrains.items()))

                if i != j:

                    ei = nMetricWithPartialAnnotationCostp.zeros((n, 1))
                    ei[i] = 1
                    ej = np.zeros((n, 1))
                    ej[j] = 1

                    p = ((ei - ej).T @ K @ (ei - ej))[0][0]

                    alpha = np.minimum(
                        lambdas[(i, j)],
                        delta
                        * self.gamma
                        * (1.0 / (p + eps) - 1.0 / (xi[(i, j)] + eps))
                        / (self.gamma + 1.0),
                    )
                    beta = delta * alpha / (1 - delta * alpha * p)
                    xi[(i, j)] = self.gamma * xi[(i, j)] / (self.gamma + delta * alpha * xi[(i, j)])
                    lambdas[(i, j)] = lambdas[(i, j)] - alpha

                    update = beta * (K @ (ei - ej) @ (ei - ej).T @ K)

                    K = K + update
                    if np.linalg.norm(update) > 1e-3:
                        upd.append(np.linalg.norm(update,ord=1))
                        convergence = True
            moyenne=np.mean(upd)
            if moyenne<1e-2:
                convergence = False
        return K

class ITML_Metric(rpt.base.BaseCost):
    """"""

    model = "Learned Kernel Partial Annotation"
    min_size = 3

    def pre_fit(
        self,
        signals,
        labels,
        upper_bound_similarity,
        lower_bound_dissimilarity,
        gamma,
        init=None,
    ):
        """computes the parameters (G_hat, G, training_samples) of the learned metrics

        Args:
            signals (List[array]): signals on which the metric is learned. List of len n_signals of
                array of shape(n_samples, n_features)
            labels (List[array]): corresponding labels of the signals. List of len n_signals of
                array of shape (n_samples, 1). The labels must be integers (>=0). The non-labelled
                samples can be anything below 0.
            upper_bound_similarity: [Bregman param] upper bound for the similarity constrains
            lower_bound_dissimilarity: [Bregman param] lower bound for the dissimilarity constrains
            gamma: [Bregman param] tradeoff between satisfying the constraints and minimizing DKL(G_hat,G)
        """
        self.u = upper_bound_similarity
        self.l = lower_bound_dissimilarity
        self.gamma = gamma

        self.training_samples, self.constrains = self.get_training_samples_and_constains(
            signals, labels
        )

        self.M = self.compute_bregman(init)

    def fit(self, signal):
        """Compute params to segment signal.
        Args:
            signal (array): signal to segment.
        """
        self.signal = signal

    def error(self, start, end):
        """Return the approximation cost on the segment [start:end].

        Args:
            start (int): start of the segment
            end (int): end of the segment

        Returns:
            segment cost
        """
        # compute equation 8.7) and then 8.8) in Charles Truong. Détection de ruptures multiples –
        # application aux signaux physiologiques.

        subsignal = self.signal[start:end]
        matrice=(subsignal@self.M).dot(subsignal.T)
        error=np.trace(matrice)-(np.sum(matrice))/(end-start)
        return error

    def _phi_m_hat_phi(self, i, j):
        ki = self.inner_product_with_training_samples[i, :][np.newaxis, :]
        kj = self.inner_product_with_training_samples[j, :][np.newaxis, :]

        return self.inner_product[i, j] + (ki @ self.G_core @ kj.T)[0][0]

    @staticmethod
    def get_training_samples_and_constains(signals, labels):
        """Derives training samples and constrains dictionnary from labels list.

        Args:
            signals (List[array]): signals on which the metric is learned. List of len n_signals of
                array of shape(n_samples, n_features)
            labels (List[array]): corresponding labels of the signals. List of len n_signals of
                array of shape (n_samples, 1). The labels must be integers (>=0).The non-labelled
                samples can be anything below 0.

        Returns:
            (list, dict): list containing the training samples and a dictionnary whose keys
                corresponds to indexes in the list and the value to the label 1 (similar) or -1
                (dissimilar)
        """
        training_samples = []
        constrains = {}
        last_idx = -1
        for signal, label in zip(signals, labels):
            begin_signal = True
            idx_max = np.max(label)
            for idx in range(idx_max + 1):

                sub_signal = signal[(label == idx).squeeze()]
                new_idx_iterator = range(last_idx + 1, sub_signal.shape[0] + last_idx + 1)

                for key in itertools.combinations(new_idx_iterator, r=2):
                    constrains[key] = 1  # similar

                if begin_signal:
                    begin_signal = False
                else:
                    for key in itertools.product(last_idx_iterator, new_idx_iterator):
                        constrains[key] = -1  # disimilar

                training_samples.append(sub_signal)
                last_idx_iterator = new_idx_iterator
                *_, last_idx = last_idx_iterator

        tr_samples=[]
        for i in training_samples:
            for j in i:
                tr_samples.append(j)

        tr_samples=np.array(tr_samples)

        return tr_samples, constrains

    def compute_bregman(self,init):
        # Algorithm 1 in P. Jain, B. Kulis, J. V. Davis, and I. S. Dhillon, “Metric and kernel
        # learning using a linear transformation,” Journal of Machine Learning Research (JMLR), vol.
        #  13, pp.519–547, 2012.

        eps = np.finfo(float).eps
        taille=self.training_samples[0].shape[0]
        print("taille",taille)
        if init.any()==None:
            K = np.eye(taille)
        else:
            K=init
        lambdas = dict.fromkeys(self.constrains.keys(), np.array(0))
        xi = dict(
            [
                (key, self.u) if value == 1 else (key, self.l)
                for key, value in self.constrains.items()
            ]
        )

        n = self.training_samples.shape[0]
        print("n egale =",n)
        convergence = True
        min_ite = int(len(self.constrains)/32)
        num_ite = 0
        while convergence:
            upd=[]
            num_ite += 1
            print(f"Iteration numéro°: {num_ite}")
            convergence = False
            for _ in range(min_ite):
                (i, j), delta = random.choice(list(self.constrains.items()))

                if i != j:
                    ei=np.array(self.training_samples[i])
                    ej=np.array(self.training_samples[j])
                    p = ((ei - ej).T @ K @ (ei - ej))

                    alpha = np.minimum(
                        lambdas[(i, j)],
                        delta
                        * (1.0 / (p + eps) - self.gamma / (xi[(i, j)] + eps))
                        / (2),
                    )
                    beta = delta * alpha / (1 - delta * alpha * p)
                    xi[(i, j)] = self.gamma * xi[(i, j)] / (self.gamma + delta * alpha * xi[(i, j)])
                    lambdas[(i, j)] = lambdas[(i, j)] - alpha
                    temp=np.expand_dims(ei-ej, axis=1)
                    update = beta * (K @ temp @ temp.T @ K)

                    K = K + update

                    if np.linalg.norm(update) > 1e-3:

                        upd.append(np.linalg.norm(update,ord=1))

                        convergence = True
            moyenne =np.linalg.norm(upd)
            print("Moyenne : ", moyenne)
            if moyenne <1e-2:
                convergence= False

        return K
