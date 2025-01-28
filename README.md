# IKFoM 
- **IKFoM** (Iterated Kalman Filters on Manifolds) is a computationally efficient and convenient toolkit for deploying iterated Kalman filters on various robotic systems, especially systems operating on high-dimension manifold. 
- Original repository - https://github.com/hku-mars/ikfom

<br>

# This repository
- Solved a few numerical issues
	- `scalar_type(1/2)` -> it is `0` at the compile step.
	- Similarly, `scalar_type(value/2)` has a problem
- `TBB` is adopted for faster parallel programming in many for loops.
- `boost::bind` of update function.
	```cpp
    esekfom::esekf<state_ikfom, 12, input_ikfom> ieskf_;
	ieskf_.init_dyn_share(func1, func2, func3,
                          boost::bind(&LodeStar::hMatrixModelShared, this, _1, _2),
                          num_max_iterations,
                          epsi);
	```