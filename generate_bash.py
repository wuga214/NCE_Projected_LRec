import argparse
from utils.io import load_yaml, get_file_names, write_file
import os


def main(args):
    bash_path = load_yaml('config/global.yml', key='path')['bashes']
    yaml_files = get_file_names('config', extension='.yml')
    #project_path = os.path.dirname(os.path.abspath(__file__))
    project_path = "~/IF-VAE-Recommendation"

    pattern = "#!/usr/bin/env bash\n" \
              "source {0}\n" \
              "cd {1}\n" \
              "python tune_parameters.py -d {2} -n {3}/{4}.csv -y config/{4}.yml\n"

    for setting in yaml_files:
        name, extension = os.path.splitext(setting)
        content = pattern.format(args.virtualenv_path, project_path, args.data_path, args.problem, name)
        write_file(bash_path+args.problem, args.problem+'-'+name+'.sh', content, exe=True)

    bash_files = sorted(get_file_names(bash_path+args.problem, extension='.sh'))

    commands = []
    command_pattern = 'sbatch --nodes=1 --time={0}:00:00 --mem={1} --cpus=4 '
    if args.gpu:
        command_pattern = command_pattern + '--gres=gpu:1 '

    command_pattern = command_pattern + '{2}'

    for bash in bash_files:
        commands.append(command_pattern.format(args.max_time, args.memory, bash))
    content = "\n".join(commands)
    write_file(bash_path + args.problem, 'run_' + args.problem + '.sh', content)



if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="CreateBash")
    parser.add_argument('-p', dest='problem', default="yahoo")
    parser.add_argument('-v', dest='virtualenv_path', default='~/ENV/bin/activate')
    parser.add_argument('-d', dest='data_path', default="data/yahoo/")
    parser.add_argument('-gpu', dest='gpu', action='store_true')
    parser.add_argument('-t', dest='max_time', default='96')
    parser.add_argument('-m', dest='memory', default='32G')


    args = parser.parse_args()

    main(args)