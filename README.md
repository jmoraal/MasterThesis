# MasterThesis
Code surrounding and/or belonging to my Master's Thesis on dislocation dynamics with creation and annihilation, mostly on regularised systems. 
Contains simulations going in various directions: 

## 1D Systems
The main research is on interacting dislocations in one dimension. There are several levels of analysis: 
* ```1DDislocationsCreation``` uses a basic manually implemented integrator, which allows for flexibility in customising. 
* ```1DautoSolver``` makes use of an automatic ODE solver, which is faster but more difficult to adapt (e.g. to creation)
* ```ODEPhaseplot``` analyses ODEs describing single creations of dipoles

## 2D Systems
Was initially tried out of curiosity, after which focusing on the 1D case (hence less elaborate and not as neat)
* ```2DDislocationsSDE``` is the main file; simulates system with sticky collisions and stochastic noise in dislocation movement. 
* ```2DDislocationsCreation``` contains beginning of dislocation creation approach; not yet finished. 
