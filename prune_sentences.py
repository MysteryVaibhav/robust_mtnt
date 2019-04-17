import sys
import os

# python prune_sentences.py data/europarl-v7.fr-en.fr data/europarl-v7-70.fr-en.fr data/europarl-v7.fr-en.en data/europarl-v7-70.fr-en.en 70

# python prune_sentences.py data/dev/newsdiscussdev2015-fren-src.fr.sgm data/dev/newsdiscussdev2015-70-fren-src.fr.sgm data/dev/newsdiscussdev2015-fren-ref.en.sgm data/dev/newsdiscussdev2015-70-fren-ref.en.sgm 70

if __name__=="__main__":
  input_1 = sys.argv[1]
  output_1 = sys.argv[2]
  input_2 = sys.argv[3]
  output_2 = sys.argv[4]
  length = int(sys.argv[5])
  with open(input_1,"r") as f1, open(output_1, "w") as f2, open(input_2,"r") as f3, open(output_2, "w") as f4:
    for line_src, line_tgt in zip(f1.readlines(), f3.readlines()):
      if len(line_src.split()) <= length:
        f2.write(line_src)
        f4.write(line_tgt)
