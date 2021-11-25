import re

from fabric import Connection
from fabric import SerialGroup as Group
import fabric
import argparse


def setup_master(master: str, passphrase: str, user: str) -> None:
    connect_kwargs = {'passphrase': passphrase}
    node = Connection(master, user=user, connect_kwargs=connect_kwargs)
    home = get_home_path(node)[0]
    # install(node)  # Not calling because of Fabric bug

    # redis conf
    logfile = f'logfile "{home}/redis.log"'
    node.put('redis_configs/redis.conf', 'redis-stable/redis.conf')
    node.run(f'echo {logfile} >> redis-stable/redis.conf')

    # sentinal conf
    logfile = f'logfile "{home}/sentinel.log"'
    sentinel_monitor = f'sentinel monitor mymaster {master} 6379 2'
    node.put('redis_configs/sentinel.conf', 'redis-stable/sentinel.conf')
    node.run(f'echo {logfile} >> redis-stable/sentinel.conf')
    node.run(f"sed -i 's/placeholder-line/{sentinel_monitor}/g' redis-stable/sentinel.conf")
    # bring_up_server(node)  # Not calling because of Fabric bug


def setup_slave(slaves: list, passphrase: str, user: str, master: str) -> None:
    connect_kwargs = {'passphrase': passphrase}
    nodes = Group(*slaves, user=user, connect_kwargs=connect_kwargs)
    home = get_home_path(nodes)[0]
    # install(nodes)  # Not calling because of Fabric bug

    for node in nodes:
        # redis conf
        logfile = f'logfile "{home}/redis.log"'
        slaveof = f'slaveof {master} 6379'
        print(logfile, slaveof)
        node.put('redis_configs/redis.conf', 'redis-stable/redis.conf')
        node.run(f'echo {slaveof} >> redis-stable/redis.conf')
        node.run(f'echo {logfile} >> redis-stable/redis.conf')

        # sentinal conf
        logfile = f'logfile "{home}/sentinel.log"'
        sentinel_monitor = f'sentinel monitor mymaster {master} 6379 2'
        print(logfile, sentinel_monitor)
        node.put('redis_configs/sentinel.conf', 'redis-stable/sentinel.conf')
        node.run(f'echo {logfile} >> redis-stable/sentinel.conf')
        node.run(f"sed -i 's/placeholder-line/{sentinel_monitor}/g' redis-stable/sentinel.conf")
    # bring_up_server(node)  # Not calling because of Fabric bug


def install(executor):
    with open('installation.sh') as f:
        executor.run(f.read())


def bring_up_server(executor):
    with open('run.sh') as f:
        executor.run(f.read())


def get_home_path(executor):
    result = executor.run('echo $HOME')
    if isinstance(result, fabric.runners.Result):
        result = {'dummy_key': result}
    out = []
    for _, val in result.items():
        regex = re.findall("/+[a-zA-Z0-9 _-]*\n", str(val))
        if len(regex) > 1:
            raise Exception('Found more than one regex match: Disastrous')
        print(regex, '--')
        out.append(regex[0][:-1])  # removing new line characters
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", help="Master host address", type=str)
    parser.add_argument("--slaves", help="Slave host addresses", type=str)
    parser.add_argument('--passphrase', help="Passphrase for ssh key file", type=str)
    parser.add_argument('--user', help="Username to the server", type=str)
    args = parser.parse_args()

    passphrase = args.passphrase
    user = args.user

    if not args.master or not args.slaves:
        raise Exception('Host argument cannot be empty')
    try:
        master = list(map(str.strip, args.master.split(',')))
    except Exception as e:
        raise Exception(f'Master argument does not have proper value: {e}')
    try:
        slaves = list(map(str.strip, args.slaves.split(',')))
    except Exception as e:
        raise Exception(f'Slave argument does not have proper value: {e}')

    if len(master) > 1:
        raise Exception('We do not support more than one master now!')
    setup_master(master[0], passphrase, user)
    setup_slave(slaves, passphrase, user, master[0])
