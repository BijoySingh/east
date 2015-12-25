from east import East
import argparse
import logging

__author__ = 'bijoy'

# Parses the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default=None, type=str, help='input to be tagged')
parser.add_argument('-s', '--sentence', default=0, type=int,
                    help=East.get_help(True, True) + " ; " + East.get_help(False, True))
parser.add_argument('-d', '--document', default=0, type=int,
                    help=East.get_help(True, False) + " ; " + East.get_help(False, False))
parser.add_argument('-m', '--sentiment', action='store_true', help='do sentiment analysis')
parser.add_argument('-c', '--club', action='store_true', help='treat a document as a single sentence')
parser.add_argument('-f', '--file', default=None, type=str, help='read input from file')
arguments = parser.parse_args()

# Evaluates the input for the analysis
cli_input = arguments.input
if arguments.file is not None and arguments.input is None:
    logging.error("No input provided, use -i or --input tag, see --help for more command line options")
    exit()
elif arguments.file is not None:
    cli_input = open(arguments.file, 'r').read()

# Creates the variable for Api
api = East(sentiment=arguments.sentiment,
           sentence_level=arguments.sentence,
           document_level=arguments.document,
           club=arguments.club)

# Prints out the result of the analysis
print(api.analyse(cli_input))
