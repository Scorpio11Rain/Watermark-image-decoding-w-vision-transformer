# Eval

To run this eval script, first make sure that the models directory containing all the trained models are in the the same directory as the output directory.

Then simply run `python eval.py`

This script produces an F1 score that evaluates how well the watermark classifier can tell the difference between watermarked and non watermarked data. It also produces an accuracy percentage that gives how accurately the decoder is able to decode bits from watermarked images.