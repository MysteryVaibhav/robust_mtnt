import argparse
import sentencepiece as spm

#MODEL_PATH = 'data/m.model'

parser = argparse.ArgumentParser()
parser.add_argument('-m','--model', help='Input file path', required=True)
parser.add_argument('-i','--input', help='Input file path', required=True)
parser.add_argument('-o','--output', help='Output file path', required=True)

args = vars(parser.parse_args())
MODEL_PATH = args['model']
input_path = args['input']
output_path = args['output']

sp = spm.SentencePieceProcessor()
sp.Load(MODEL_PATH)

with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
	for line in infile:
		pieces = line.split(' ')
		decoded = sp.DecodePieces(pieces)
		outfile.write(decoded)
