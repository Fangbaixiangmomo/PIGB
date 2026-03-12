# Log

This is a research diary for my undergrad thesis `Physics-Informed Gradient Boosting`.

## Mar. 12th, 2026

Ran all existing methods on 1d bs eqn today. Traditional methods seems nice. Seems that PINN is glossing over why they didnt report MSE in the paper. I didn't read it thoroughly yet. Maybe it's worth doing it tomorrow.

The preliminary PIGB is terrible! Why?

Wait! Trees are bad... but trees with depth 1 is really good! That actually makes sense. it's just piecewise constant. Maybe this tells us something. We should switch to smoother learners.

Oh my fucking lord!!! Ran it on a 3-spline with 16 knots. The result is fantastic! Although you will need to deal with early-stopping, but that can be removed once you added moving average...
