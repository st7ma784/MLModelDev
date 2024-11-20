
import os,sys


#### This is our launch function, which builds the dataset, and then runs the model on it.


def train(config={
        "batch_size":16, # ADD MODEL ARGS HERE
         "codeversion":"-1",
    },dir=None,devices=None,accelerator=None,Dataset=None,logtool=None,EvalOnLaunch=False):

    import pytorch_lightning
    from pytorch_lightning.callbacks import TQDMProgressBar,EarlyStopping
    import datetime
    from pytorch_lightning.strategies import FSDPStrategy
    from pytorch_lightning.plugins.environments import SLURMEnvironment

    #### EDIT HERE FOR DIFFERENT VERSIONS OF A MODEL
    from models.train import myLightningModule

    model=myLightningModule(  **config)
    if dir is None:
        dir=config.get("dir",".")
    if Dataset is None:
        from DataModule import MyDataModule
        Dataset=MyDataModule(Cache_dir=dir,**config)
    if devices is None:
        devices=config.get("devices","auto")
    if accelerator is None:
        accelerator=config.get("accelerator","auto")
    # print("Training with config: {}".format(config))
    Dataset.batch_size=config["batch_size"]
    filename="model-{}-{}".format(config["codeversion"],config["batch_size"])
    callbacks=[
        TQDMProgressBar(),
        EarlyStopping(monitor="train_loss", mode="min",patience=10,check_finite=True,stopping_threshold=0.001),
        #save best model
        pytorch_lightning.callbacks.ModelCheckpoint(
            monitor='train_loss',
            dirpath=dir,
            filename=filename,
            save_top_k=1,
            mode='min',
            save_last=True,),
    ]
    p=config['precision']
    if isinstance(p,str):
        p=16 if p=="bf16" else int(p)  ##needed for BEDE
    print("Launching with precision",p)

    #workaround for NCCL issues on windows
    if sys.platform == "win32":
        os.environ["PL_TORCH_DISTRIBUTED_BACKEND"]='gloo'
    EVALOnLaunch=config.get("EVALOnLaunch",EVALOnLaunch)
    trainer=pytorch_lightning.Trainer(
            devices=2 if not EVALOnLaunch else 1,
            num_nodes=3 if not EVALOnLaunch else 1,
            accelerator="gpu",
            max_epochs=200 if not EVALOnLaunch else 1,
            #profiler="advanced",
            #plugins=[SLURMEnvironment()],
            #https://lightning.ai/docs/pytorch/stable/clouds/cluster_advanced.html
            logger=logtool,
            strategy=FSDPStrategy(accelerator="gpu",
                                   parallel_devices=6 if not EVALOnLaunch else 1,
                                   cluster_environment=SLURMEnvironment(),
                                   timeout=datetime.timedelta(seconds=1800),
                                   #cpu_offload=True,
                                   #mixed_precision=None,
                                   #auto_wrap_policy=True,
                                   #activation_checkpointing=True,
                                   #sharding_strategy='FULL_SHARD',
                                   #state_dict_type='full'
            ),
            callbacks=callbacks,
            gradient_clip_val=0.25,# Not supported for manual optimization
            fast_dev_run=False,
            precision=p
    )
    if EVALOnLaunch:
        #load model from pytorch lightning checkpoint
        model=model.load_from_checkpoint(filepath=os.path.join(dir,filename+".ckpt"))
        trainer.test(model,Dataset)
    else:
        trainer.fit(model,Dataset)
    

#### This is a wrapper to make sure we log with Weights and Biases, You'll need your own user for this.
def wandbtrain(config=None,dir=None,devices=None,accelerator=None,Dataset=None):

    import pytorch_lightning
    if config is not None:
        config=config.__dict__
        dir=config.get("dir",dir)
        logtool= pytorch_lightning.loggers.WandbLogger( project="TestDeploy",entity="st7ma784", save_dir=dir)
        print(config)

    else:
        #We've got no config, so we'll just use the default, and hopefully a trainAgent has been passed
        import wandb
        print("Would recommend changing projectname according to config flags if major version swithching happens")
        run=wandb.init(project="TestDeploy",entity="st7ma784",name="TestDeploy",config=config)
        logtool= pytorch_lightning.loggers.WandbLogger( project="TestDeploy",entity="st7ma784",experiment=run, save_dir=dir)
        config=run.config.as_dict()

    train(config,dir,devices,accelerator,Dataset,logtool)


def neptunetrain(config=None,dir=None,devices=None,accelerator=None,Dataset=None):
    
        import pytorch_lightning
        if config is not None:
            config=config.__dict__
            dir=config.get("dir",dir)
            logtool= pytorch_lightning.loggers.NeptuneLogger( project="TestDeploy",entity="st7ma784", save_dir=dir)
            print(config)
    
        else:
            #We've got no config, so we'll just use the default, and hopefully a trainAgent has been passed
            import neptune
            print("Would recommend changing projectname according to config flags if major version swithching happens")
            run=neptune.init(project="TestDeploy",entity="st7ma784",name="TestDeploy",config=config)
            logtool= pytorch_lightning.loggers.NeptuneLogger( project="TestDeploy",entity="st7ma784",experiment=run, save_dir=dir)
            config=run.config.as_dict()
    
        train(config,dir,devices,accelerator,Dataset,logtool)


def SLURMEval(ModelPath,config):
    job_with_version = '{}v{}'.format("EVAL", 0)
    sub_commands =['#!/bin/bash',
        '# Auto-generated by test-tube (
        '#SBATCH --time={}'.format( '24:00:00'),# Max run time
        '#SBATCH --job-name={}'.format(job_with_version),
        '#SBATCH --nodes=1',
        '#SBATCH --ntasks-per-node=1',
        '#SBATCH --gres=gpu:1',
        '#SBATCH --signal=USR1@{}'.
        '#SBATCH --mail-type={}'.format(','.join(['END','FAIL'])),
        '#SBATCH --mail-user={}'.format(<YOURMAIL>),
  ]
    comm="python"
    slurm_commands={}

    if str(os.getenv("HOSTNAME","localhost")).endswith("bede.dur.ac.uk"):
        sub_commands.extend([
                '#SBATCH --account MYACOCUNT',
                'export CONDADIR=/nobackup/projects/<BEDEPROJECT>/$USER/miniconda',
                'export NCCL_SOCKET_IFNAME=ib0'])
        comm="python3"
    else:

        sub_commands.extend(['#SBATCH -p gpu-medium',
                             'export CONDADIR=/storage/hpc/46/manders3/conda4/open-ce',
                             'export NCCL_SOCKET_IFNAME=enp0s31f6',])
    sub_commands.extend([ '#SBATCH --{}={}\n'.format(cmd, value) for  (cmd, value) in slurm_commands.items()])
    sub_commands.extend([
        'export SLURM_NNODES=$SLURM_JOB_NUM_NODES',
        'export wandb=9cf7e97e2460c18a89429deed624ec1cbfb537bc',
        'source /etc/profile',
        'module add opence',
        'conda activate $CONDADIR',# ...and activate the conda environment
    ])




def SlurmRun(trialconfig):

    job_with_version = '{}v{}'.format("SINGLEGPUTESTLAUNCH", 0)

    sub_commands =['#!/bin/bash',
        '# Auto-generated by test-tube (https://github.com/williamFalcon/test-tube)',
        '#SBATCH --time={}'.format( '24:00:00'),# Max run time
        '#SBATCH --job-name={}'.format(job_with_version),
        '#SBATCH --nodes=2',  #Nodes per experiment
        '#SBATCH --ntasks-per-node=3',# Set this to GPUs per node.
        '#SBATCH --gres=gpu:3',  #{}'.format(per_experiment_nb_gpus),
        f'#SBATCH --signal=USR1@{5 * 60}',
        '#SBATCH --mail-type={}'.format(','.join(['END','FAIL'])),
        '#SBATCH --mail-user={}'.format('YOURMAIL@gmail.com'),
    ]
    comm="python"
    slurm_commands={}

    if str(os.getenv("HOSTNAME","localhost")).endswith("bede.dur.ac.uk"):
        sub_commands.extend([
                '#SBATCH --account MYACOCUNT',
                '''
                arch=$(uname -i) # Get the CPU architecture
                if [[ $arch == "aarch64" ]]; then
                   # Set variables and source scripts for aarch64
                   export CONDADIR=/nobackup/projects/<project>/$USER/ # Update this with your <project> code.
                   source $CONDADIR/aarchminiconda/etc/profile.d/conda.sh
                fi
                '''
                'export CONDADIR=/nobackup/projects/<BEDEPROJECT>/$USER/miniconda',
                'export NCCL_SOCKET_IFNAME=ib0'])
        comm="python3"
    else:

        sub_commands.extend(['#SBATCH -p gpu-medium',
                             'export CONDADIR=/storage/hpc/46/manders3/conda4/open-ce',
                             'export NCCL_SOCKET_IFNAME=enp0s31f6',])
    sub_commands.extend([ '#SBATCH --{}={}\n'.format(cmd, value) for  (cmd, value) in slurm_commands.items()])
    sub_commands.extend([
        'export SLURM_NNODES=$SLURM_JOB_NUM_NODES',
        'export wandb=9cf7e97e2460c18a89429deed624ec1cbfb537bc',
        'source /etc/profile',
        'module add opence',
        'conda activate $CONDADIR',# ...and activate the conda environment
    ])
    script_name= os.path.realpath(sys.argv[0]) #Find this scripts name...
    trialArgs=__get_hopt_params(trialconfig)
    #If you're deploying prototyping code and often changing your pip env,
    # consider adding in a 'scopy requirements.txt
    # and then append command 'pip install -r requirements.txt...
    # This should add your pip file from the launch dir to the run location, then install on each node.

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
    from demoparse import parser
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
        
        if os.getenv("WANDB_API_KEY"):

            wandbtrain(hyperparams)
        elif os.getenv("NEPTUNE_API_TOKEN"):
            print("NEPTUNE API KEY found")
            neptunetrain(hyperparams)
        else:
            print("No logging API found, using default config")
            train(hyperparams)
    #OR To run with Default Args
    else:
        # check for wandb login details in env vars
        if os.getenv("WANDB_API_KEY"):
            print("WANDB API KEY found")
            trials= myparser.generate_wandb_trials(<WANDBUSER>,<YOURPROJECTNAME>)
        #check for neptune login details in env vars
        elif os.getenv("NEPTUNE_API_TOKEN"):
            print("NEPTUNE API KEY found")
            trials= myparser.generate_neptune_trials(<NEPTUNEUSER>,<YOURPROJECTNAME>))
        else:
            print("No logging API found, using default config")
            trials= hyperparams.generate_trials(NumTrials)  
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
