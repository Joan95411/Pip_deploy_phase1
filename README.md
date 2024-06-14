
## How to start the PIPNet webapp
- Create conda env with `conda env create -f environment.yml`
- Start MySQL server 
- Input MySQL credentials in `backend/PukkaJListener.py/SQLCredentials.` and `frontend/SQLCredentials.py`.
- Run `database/db-schema.sql` to create database 
- Place the model inside the main directory, with the path `deplpip/model`. PukkaJ Listener will only load the model 
if it is present in this directory. If you want to change the model path, change it in 
`backend/PukkaJListener.py/MODEL_DIR`.
- Read `pukkaJ-in/README.md` for more details on what's the input format.  
- Run `backend/PukkaJListener.py` to start listener. It will listen on pukkaJ-in directory.
- Run `frontend/webapp.py` to start webapp.



Sidenote: Before the model was in the directory `runs/troch_testrun_seed280493`. You will need to rename 
this `troch_testrun_seed280493` folder to `model` and place it in the main directory. The reason for this is that GitHub 
cannot handle too large files like the model. 