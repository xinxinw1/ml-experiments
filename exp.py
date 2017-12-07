#!/usr/bin/python

import argparse
import json
import pprint
import subprocess

with open('experiments.json', 'r') as f:
    experiments_json = json.load(f)
    experiments = {}
    for experiment in experiments_json:
        experiments[(experiment['name'], experiment['tag'])] = experiment

def exec_list(args):
    format_str = '%-15s %-20s %-25s %s'
    print(format_str % ('name', 'tag', 'commit', 'server'))
    for experiment in experiments_json:
        print(format_str % (experiment['name'], experiment['tag'], experiment['commit'], experiment['server']))

def exec_run(args):
    tag = args.tag
    name = args.name
    experiment = experiments[(name, tag)]
    commit = experiment['commit']
    server = experiment['server']
    subprocess.run(['./server/run_on_server', name, tag, commit, server])

def exec_setup(args):
    tag = args.tag
    name = args.name
    experiment = experiments[(name, tag)]
    commit = experiment['commit']
    server = experiment['server']
    subprocess.run(['./server/setup_server', name, tag, commit, server])

def exec_upload(args):
    tag = args.tag
    name = args.name
    experiment = experiments[(name, tag)]
    commit = experiment['commit']
    server = experiment['server']
    subprocess.run(['./server/upload_to_server', name, tag, commit, server])

def exec_continue(args):
    tag = args.tag
    name = args.name
    experiment = experiments[(name, tag)]
    server = experiment['server']
    subprocess.run(['./server/continue_on_server', name, tag, server])

def exec_download(args):
    tag = args.tag
    name = args.name
    experiment = experiments[(name, tag)]
    server = experiment['server']
    subprocess.run(['./server/download_from_server', name, tag, server])

def exec_clean_remote(args):
    tag = args.tag
    name = args.name
    experiment = experiments[(name, tag)]
    server = experiment['server']
    subprocess.run(['./server/clean_remote', name, tag, server])

def exec_clean_local(args):
    tag = args.tag
    name = args.name
    subprocess.run(['./server/clean_local', name, tag])

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()

subparser = subparsers.add_parser('list')
subparser.set_defaults(func=exec_list)

subparser = subparsers.add_parser('setup')
subparser.add_argument('name')
subparser.add_argument('tag')
subparser.set_defaults(func=exec_setup)

subparser = subparsers.add_parser('upload')
subparser.add_argument('name')
subparser.add_argument('tag')
subparser.set_defaults(func=exec_upload)

subparser = subparsers.add_parser('run')
subparser.add_argument('name')
subparser.add_argument('tag')
subparser.set_defaults(func=exec_run)

subparser = subparsers.add_parser('continue')
subparser.add_argument('name')
subparser.add_argument('tag')
subparser.set_defaults(func=exec_continue)

subparser = subparsers.add_parser('download')
subparser.add_argument('name')
subparser.add_argument('tag')
subparser.set_defaults(func=exec_download)

subparser = subparsers.add_parser('clean-remote')
subparser.add_argument('name')
subparser.add_argument('tag')
subparser.set_defaults(func=exec_clean_remote)

subparser = subparsers.add_parser('clean-local')
subparser.add_argument('name')
subparser.add_argument('tag')
subparser.set_defaults(func=exec_clean_local)

args = parser.parse_args()
args.func(args)
