import os
import subprocess 



frame_types = ['Issue-General','Issue-Specific','Issue-Specific-Combined','Narrative','all_frames']

device = 2
num_epochs = 60
thresh_setting = 'default'
date = '11-05-20'
#seeds = [12,23,35,42,45]
seeds = [45]
for seed in seeds:
	for frame_type in frame_types:
		train_file = f'/shared/2/projects/framing/data/labeled_data/dataset_11-03-20/roberta/train/{frame_type}.tsv'
		eval_file = f'/shared/2/projects/framing/data/labeled_data/dataset_11-03-20/roberta/dev/{frame_type}.tsv'
		#lmpath = '/shared/2/projects/framing/models/finetune/roberta_cased_09-01-20'
		lmpath = 'roberta-base'
		output_dir = f'/shared/2/projects/framing/models/classify/{frame_type}/roberta_baseline_{date}_{num_epochs}_epochs_{thresh_setting}_thresh_{seed}_seed'

		command = f"python roberta_classify.py \
					--lmpath {lmpath} \
					--train-file {train_file} \
					--eval-file {eval_file} \
					--output-dir {output_dir} \
					--num-epochs {num_epochs} \
					--thresh-setting {thresh_setting} \
					--manual-seed {seed} \
					--device {device} "

		# check if output dir exists and contains config.json
		if os.path.exists(output_dir) and ('config.json' in os.listdir(output_dir)):
			print('COMPLETED: ', seed, frame_type, output_dir)
		else:
			print('NOT COMPLETE: ', seed, frame_type, output_dir)
			subprocess.run(command,shell=True)