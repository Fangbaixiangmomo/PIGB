# Log

This is a research diary for my undergrad thesis `Physics-Informed Gradient Boosting`.

## Mar. 12th, 2026

Ran all existing methods on 1d bs eqn today. Traditional methods seems nice. Seems that PINN is glossing over why they didnt report MSE in the paper. I didn't read it thoroughly yet. Maybe it's worth doing it tomorrow.

The preliminary PIGB is terrible! Why?

Wait! Trees are bad... but trees with depth 1 is really good! That actually makes sense. it's just piecewise constant. Maybe this tells us something. We should switch to smoother learners.

Oh my fucking lord!!! Ran it on a 3-spline with 16 knots. The result is fantastic! Although you will need to deal with early-stopping, but that can be removed once you added moving average...

## Mar. 12th, 2026 (cont.)

Aha. I think I finally see the recursion a bit more cleanly now.

The original PI-GB update is not just “fit a tree to some weird pseudo-response.” There is actually an ensemble recursion hiding inside it. Let $F_m$ denote the predictor after $m$ boosting steps, and let $\mathcal L$ be the Black–Scholes operator

$$
\mathcal L[F]=\partial_t F+\frac12 \sigma^2 S^2 \partial_{SS}F+rS\partial_S F-rF.
$$

Here $S$ is the asset price, $t$ is time, $r$ is risk-free rate, and $\sigma$ is volatility.

In the interior, the physics pseudo-target is

$$
y_{\mathrm{phys}} = F_m - \alpha \mathcal L[F_m],
$$

where $\alpha > 0$ is the physics step size. So the residual that the stage learner is trying to fit is really

$$
y_{\mathrm{phys}} - F_m = -\alpha \mathcal L[F_m].
$$

If $\Pi_{\mathcal H}$ denotes projection onto the learner class $\mathcal H$ (say trees, splines, whatever weak learner family we use), and $\nu > 0$ is the boosting learning rate, then the idealized stagewise ensemble update is

$$
F_{m+1}=F_m + \nu \Pi_{\mathcal H}\bigl(-\alpha \mathcal L[F_m]\bigr).
$$

But now comes the fun part. If instead of raw boosting we do Boulevard-style averaging, then define the averaged ensemble by

$$
F_m = \frac{1}{m}\sum_{j=1}^m \nu h_j,
$$

where $h_j$ is the $j$-th fitted weak learner. Then the recursion becomes

$$
F_{m+1}=\frac{m}{m+1}F_m+\frac{1}{m+1}\nu \Pi_{\mathcal H}\bigl(-\alpha \mathcal L[F_m]\bigr).
$$

This is actually very clean. Rewriting,

$$
F_{m+1}=F_m+\frac{1}{m+1}\left[\nu \Pi_{\mathcal H}\bigl(-\alpha \mathcal L[F_m]\bigr) - F_m\right].
$$

This means the thing is not merely some annoying training trajectory anymore. It is trying to approach a fixed point.

If a limit $F_\infty$ exists, then it should satisfy

$$
F_\infty=\nu \Pi_{\mathcal H}\bigl(-\alpha \mathcal L[F_\infty]\bigr).
$$

This is kind of a big deal. It means adding moving average is not just a numerical hack to calm down boosting. It gives an early-stopping-free algorithmic target. Very very nice.

Now, one slightly awkward thing: if I initialize at zero and only keep this interior physics equation, then after linearizing $\Pi_{\mathcal H}$ into some linear operator $P$, the fixed point equation becomes

$$
F_\infty = \nu P\bigl(-\alpha \mathcal L[F_\infty]\bigr),
$$

or equivalently

$$
\bigl(I + \nu\alpha P\mathcal L\bigr)F_\infty = 0.
$$

That is homogeneous. So unless the operator has a nontrivial null space, the fixed point is just $0$. Oops. Slight issue. So the nontriviality of the limit must come from somewhere else.

This is where I think the prior / lifting picture finally makes sense. For option pricing PDEs, the terminal payoff and the boundary conditions are actually known from the contract and no-arbitrage logic. So instead of pretending I have some magical prior on the full solution, I only need a lifting function $\phi(S,t)$ that satisfies the terminal and boundary conditions. Then write

$$
V = \phi + G,
$$

where $V$ is the option price surface and $G$ is the residual correction.

For Black–Scholes call, the canonical conditions are

$$
V(S,T) = (S-K)^+,\qquad V(0,t)=0,
$$

and as $S\to\infty$,

$$
V(S,t)\sim S-Ke^{-r(T-t)}.
$$

So I choose $\phi$ to satisfy these, and let $G$ absorb the rest. Plugging into the PDE $\mathcal L[V]=0$ gives

$$
\mathcal L[G+\phi] = 0\quad\Longrightarrow\quad\mathcal L[G] = -\mathcal L[\phi].
$$

Then the Boulevardized recursion should really be written on $G$:

$$
G_{m+1}=\frac{m}{m+1}G_m+\frac{1}{m+1}\nu\Pi_{\mathcal H}\bigl(-\alpha(\mathcal L[G_m]+\mathcal L[\phi])\bigr).
$$

And the corresponding fixed point becomes

$$
G_\infty=\nu\Pi_{\mathcal H}\bigl(-\alpha(\mathcal L[G_\infty]+\mathcal L[\phi])\bigr),\qquad V_\infty = \phi + G_\infty.
$$

This is much better. Much much better. Because now the nontrivial part comes from the actual PDE specification, not from some leap-of-faith prior. In option pricing, terminal and boundary conditions are available at least. That makes this route actually defensible.

Another nice thing: if $\Pi_{\mathcal H}$ behaves like some linear operator $P$, then

$$
\bigl(I+\nu\alpha P\mathcal L\bigr)G_\infty=-\nu\alpha P\mathcal L[\phi].
$$

So the limit solves a linear operator equation. This is cool for at least two reasons:

First, you get rid of early stopping.

Second, if $P$ obeys some nice law asymptotically, maybe there is inference to be done on the limiting object. Of course that is a whole other can of worms. Need to specify what $P$ even is, in what sense it converges, what randomness is left, etc. But at least there is a target now.

The downside is also very clear now. If the limit is basically

$$
G_\infty=-\bigl(I+\nu\alpha P\mathcal L\bigr)^{-1}\nu\alpha P\mathcal L[\phi],
$$

then the final estimator is still a structured transformation of the lifting $\phi$. So we traded early-stopping dependence for lifting dependence. For option pricing PDE this is probably okay, because the terminal and boundary conditions are known anyway. But for a general PDE with no analytical solution and no canonical lifting, the choice of $\phi$ could still be a leap of faith.
