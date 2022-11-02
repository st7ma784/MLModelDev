
import pytorch_lightning
from pytorch_lightning.callbacks import TQDMProgressBar,EarlyStopping
import os,sys

#### This is our launch function, which builds the dataset, and then runs the model on it. 
def train(config={
        "batch_size":16, # ADD MODEL ARGS HERE
         "codeversion":"-1",        
    },dir=None,devices=None,accelerator=None,Dataset=None,logtool=None):
    
    #### EDIT HERE FOR DIFFERENT VERSIONS OF A MODEL
     if codeversion==1:
        from models.train import myLightningModule
    elif codeversion==2:
        from models.trainv2 import myLightningModule
    elif codeversion==3:
        from models.trainv3 import myLightningModule
    elif codeversion==4:
        from models.trainv4 import myLightningModule
    elif codeversion==5:
        from models.trainv5 import myLightningModule
    elif codeversion==6:
        from models.trainv6 import myLightningModule
    else:
        print("NO CODE VER SPECIFIED!!")
        from models.train import myLightningModule

    model=LightningT5Module(  **config)
    if dir is None:
        dir=config.get("dir",".")
    if Dataset is None:
        from DataModule import *
        Dataset=MyDataModule(Cache_dir=dir,**config)
    if devices is None:
        devices=config.get("devices","auto")
    if accelerator is None:
        accelerator=config.get("accelerator","auto")
    # print("Training with config: {}".format(config))
    Dataset.batch_size=config["batch_size"]
    callbacks=[
        TQDMProgressBar(),
        EarlyStopping(monitor="train_loss", mode="min",patience=10,check_finite=True,stopping_threshold=0.001),
    ]
    p=config['precision']
    if isinstance(p,str):
        p=16 if p=="bf16" else int(p)  ##needed for BEDE
    print("Launching with precision",p)

    #workaround for NCCL issues on windows 
    if sys.platform == "win32":
        os.environ["PL_TORCH_DISTRIBUTED_BACKEND"]='gloo'
    trainer=pytorch_lightning.Trainer(
            devices=devices,
            # auto_select_gpus=True,
            accelerator=accelerator,
            max_epochs=200,
            #profiler="advanced",
            logger=logtool,
            strategy="ddp",
            num_nodes=int(os.getenv("SLURM_NNODES",1)),
            callbacks=callbacks,
            gradient_clip_val=0.25,# Not supported for manual optimization
            fast_dev_run=False,
            precision=p
    )
    if config["batch_size"] !=1:
        
        trainer.fit(model,Dataset)
    else:
        return 0 #No need to train if batch size is 1
    
#### This is a wrapper to make sure we log with Weights and Biases, You'll need your own user for this. 
def wandbtrain(config=None,dir=None,devices=None,accelerator=None,Dataset=None):
    if config is not None:
        config=config.__dict__
        dir=config.get("dir",dir)
        logtool= pytorch_lightning.loggers.WandbLogger( project="<PROJECTNAME>",entity="<WANDBUSER>", save_dir=dir)
        print(config)

    else: 
        #We've got no config, so we'll just use the default, and hopefully a trainAgent has been passed
        import wandb
        print("here")
        run=wandb.init(project="<PROJECTNAME>",entity="<WANDBUSER>",name="<PROJECTNAME>",config=config)
        logtool= pytorch_lightning.loggers.WandbLogger( project="<PROJECTNAME>",entity="<WANDBUSER>",experiment=run, save_dir=dir)
        config=run.config.as_dict()
    
    train(config,dir,devices,accelerator,Dataset,logtool)
def SlurmRun(trialconfig):

    job_with_version = '{}v{}'.format("SINGLEGPUTESTLAUNCH", 0)

    sub_commands =['#!/bin/bash',
        '# Auto-generated by test-tube (https://github.com/williamFalcon/test-tube)',   
        '#SBATCH --time={}'.format( '24:00:00'),# Max run time
        '#SBATCH --job-name={}'.format(job_with_version), 
        '#SBATCH --nodes=1',  #Nodes per experiment
        '#SBATCH --ntasks-per-node=1',# Set this to GPUs per node. 
        '#SBATCH --gres=gpu:1',  #{}'.format(per_experiment_nb_gpus),
        f'#SBATCH --signal=USR1@{5 * 60}',
        '#SBATCH --mail-type={}'.format(','.join(['END','FAIL'])),
        '#SBATCH --mail-user={}'.format('YOURMAIL@gmail.com'),
    ]
    comm="python"
    if str(os.getenv("HOSTNAME","localhost")).endswith("bede.dur.ac.uk"):
        sub_commands.extend(['export CONDADIR=/nobackup/projects/<BEDEPROJECT>/$USER/miniconda',
                            'export NCCL_SOCKET_IFNAME=ib0'])
        slurm_commands={"account":"<BEDEPROJECT>"}#,"partition":"gpu"} Leaving this part out to run on non-bede slurm
        comm="python3"
    else: 
        sub_commands.extend(['export CONDADIR=/home/$USER/miniconda3',
                             'export NCCL_SOCKET_IFNAME=enp0s31f6',])
        slurm_commands={}
    sub_commands.extend([ '#SBATCH --{}={}\n'.format(cmd, value) for  (cmd, value) in slurm_commands.items()])
    sub_commands.extend([
        'export SLURM_NNODES=$SLURM_JOB_NUM_NODES',
        'export wandb=<YOURWANDBAPIKEY>',
        'source $CONDADIR/etc/profile.d/conda.sh',
        'conda activate $CONDADIR/envs/open-ce',# ...and activate the conda environment
    ])
    script_name= os.path.realpath(sys.argv[0]) #Find this scripts name...
    trialArgs=__get_hopt_params(trialconfig)

    sub_commands.append('srun {} {} {}'.format(comm, script_name,trialArgs))
    #when launched, this script will be called with no trials, and so drop into the wandbtrain section, 
    sub_commands = [x.lstrip() for x in sub_commands]        

    full_command = '\n'.join(sub_commands)
    return full_command

def __get_hopt_params(trial):
    """
    Turns hopt trial into script params
    :param trial:
    :return:
    """
    params = []
    for k in trial.__dict__:
        v = trial.__dict__[k]
        if k == 'num_trials':
            v=0
        # don't add None params
        if v is None or v is False:
            continue

        # put everything in quotes except bools
        if __should_escape(v):
            cmd = '--{} \"{}\"'.format(k, v)
        else:
            cmd = '--{} {}'.format(k, v)
        params.append(cmd)

    # this arg lets the hyperparameter optimizer do its thin
    full_cmd = ' '.join(params)
    return full_cmd

def __should_escape(v):
    v = str(v)
    return '[' in v or ';' in v or ' ' in v
if __name__ == '__main__':
    from argParser import parser
    from subprocess import call

    myparser=parser()
    hyperparams = myparser.parse_args()

    defaultConfig=hyperparams.__dict__
    
    NumTrials=hyperparams.num_trials
    #BEDE has Env var containing hostname  #HOSTNAME=login2.bede.dur.ac.uk check we arent launching on this node
    if NumTrials==-1:
        #debug mode - We want to just run in debug mode... 
        #pick random config and have at it! 
        
        trial=hyperparams.generate_trials(1)[0]
        #We'll grab a random trial, BUT have to launch it with KWARGS, so that DDP works.       
        #result = call('{} {} --num_trials=0 {}'.format("python",os.path.realpath(sys.argv[0]),__get_hopt_params(trial)), shell=True)

        print("Running trial: {}".format(trial))
        
        wandbtrain(trial)

    elif NumTrials ==0 and not str(os.getenv("HOSTNAME","localhost")).startswith("login"): #We'll do a trial run...
        #means we've been launched from a BEDE script, so use config given in args///
        wandbtrain(hyperparams)

    #OR To run with Default Args
    else: 
        trials=hyperparams.generate_trials(NumTrials)

        for i,trial in enumerate(trials):             
            command=SlurmRun(trial)
            slurm_cmd_script_path =  os.path.join(defaultConfig.get("dir","."),"slurm_cmdtrial{}.sh".format(i))

            with open(slurm_cmd_script_path, "w") as f:
                f.write(command)
            print('\nlaunching exp...')
            result = call('{} {}'.format("sbatch", slurm_cmd_script_path), shell=True)
            if result == 0:
                print('launched exp ', slurm_cmd_script_path)
            else:
                print('launch failed...')  
