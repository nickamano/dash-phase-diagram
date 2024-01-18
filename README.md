# Convex Hull in 3D

This is a project for generation a plotly dash dashboard containing the binary regular solution Gibbs free energy curves and convex hull in 3D. 

# How to run

you can run the dash applet using the following commands. This is creating a new python environment for the applicaiton. If you would not like to do that and already have all required packages, just run the last line.
```
python3 -m venv env
source env/bin/activate
pip install numpy scipy plotly dash
python main.py
```

I have also included a `requirements.txt` file if you would rather create your python environment with that file 

# How to use Docker

You can run the code in a docker container using the following code.

```
docker compose up --detach
```

This will run the application on local port `2222`. I recommend running the application detached as the debug is cluttered with numpy warnings. 