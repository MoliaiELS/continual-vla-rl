@{
    SshHost = 'idac_sever'
    Repository = 'https://github.com/MoliaiELS/continual-vla-rl.git'
    Branch = 'fyp/deploy-base'
    CodeRoot = '/mnt/slurmfs-4090node1/homes/yguo704/RLinf'
    RunCodeRoot = '/mnt/slurmfs-4090node1/homes/yguo704/RLinf-runs'
    DataRoot = '/mnt/slurmfs-4090node3/user_data/yguo704/RLinf'
    CondaRoot = '/mnt/slurmfs-4090node1/homes/yguo704/miniforge3'
    Partition = 'GPU'
    Node = '' # Slurm chooses an available compute node matching Gres.
    Gres = 'gpu:4090:1'
    Cpus = 8
}
