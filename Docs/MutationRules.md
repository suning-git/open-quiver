The goal is to generate a framing of a quiver and the mutation operation.

A **quiver** $Q=(Q_0,Q_1,s,t)$ consists of following data:
- A set $Q_0$ whose elements are called **vertices**. We often assume that $Q_0=[n]\coloneqq\{1,2,\dots,n\}$ and denote its vertices by $i,j$, etc.
- A set $Q_1$ with two maps $s,t\colon Q_1 \to Q_0$, such that for any $\alpha \in Q_1$ we have $s(\alpha) \overset{\alpha}{\to} t(\alpha)$. Its elemets are called **arrows**. The data $s,t$ are often ommited.

We always require that $Q$ is 2-acyclic, i.e., does not contain loop or 2-cycles. Here
- A Loop is an arrow of the form $i \to i$.
- A 2-cycles is a pair of arrows of the form $i\to j$, $j\to i$.

An ice quiver is a pair $(Q,F)$ where $Q$ is a quiver, and $F\subset Q_0$ is a set of **frozen vertices** with the restriction that any $i,j \in F$ there is no arrows of $Q$ connecting them. By convention, we assume $Q_0\setminus F=[n]$ and $F=[n+1,m]\coloneqq \{n+1,n+2,\dots,m\}$. We call elements of $Q_0\setminus F$ as **mutable vertices**.

The **mutation** of an 2-acyclic ice quiver $(Q,F)$ at a mutable vertex $k$, denoted by $\mu_k$, produces a new ice quiver $(\mu_kQ,F)$ by the three steps process:
1. For every 2-path $i\to k \to j$, add a new arrow $i \to j$.
2. Reverse the direction of all arrows incident to $k$ in $Q$.
3. Remove any 2-cycles created, and remove any arrows created that connect two frozen vertices.

We denote the set of all ice quivers that created during this process as $\mathrm{Mut}((Q,F))$

For a quiver $Q$, its **framed** (resp., **coframed**) quiver $\widehat{Q}$ (resp., $\widecheck{Q}$) is defined as follows:
- Its vertices are $\widehat{Q}_0=Q_0\sqcup [n+1,2n]$. Here we assume that $Q_0=[n]$.
- Its arrows are
$$
\widehat{Q}_1=Q_1 \sqcup \{i \to n+i\colon i \in [n]\}
$$
(resp., $\widecheck{Q}_1=Q_1 \sqcup \{n+i \to i\colon i \in [n]\}$).

We often denote vertex $i+n$ as $i'$ for $i \in [n]$.

Now consider all mutation on a framed quiver $\widehat{Q}$, and for an ice quiver $\overline{Q} \in \mathrm{Mut}(\widehat{Q})$. A mutable vertex $i \in \overline{Q}_0$ is **green** (resp., **red**) if there are no arrows in $\overline{Q}$ of the form $j' \to i$ (resp., $i \to j'$) for some $j \in [n]$.

An isomorphism $\varphi$ between two quiver $Q,Q'$ are two bijections $\varphi_0\colon Q_0 \to Q_0'$, $\varphi_1\colon Q_1 \to Q_1'$ such that
$$
\varphi_0 \circ s = s' \circ \varphi_1,\quad \varphi_0 \circ t = t' \circ \varphi_1.
$$


Here are two important theorems. Say $\widehat{Q}$ is the framed quiver of $Q$.
1. Any vertex $i$ of $\overline{Q} \in \mathrm{Mut}(\widehat{Q})$ is either green or red.
2. If a $\overline{Q} \in \mathrm{Mut}(\widehat{Q})$ satisfies all vertices are red, then there is an isomorphism $\varphi$ from $\overline{Q}$ to the coframed quiver $\widecheck{Q}$ of $Q$, and $\varphi_0(F)=F$.


For an ice quiver $(Q,F)$, its **exchange matrix** is an $n\times m$ matrix with entry
$$
b_{ij} = |\{i\overset{\alpha}{\to} j \in Q_1\}| - |\{j\overset{\alpha}{\to} i \in Q_1\}|.
$$