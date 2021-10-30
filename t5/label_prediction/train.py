import wandb
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch
from torch.utils.data import Dataset 
import argparse 
import random

#FIX WANBD PART
# https://docs.wandb.ai/guides/integrations/huggingface
%env WANDB_PROJECT=knowledge-discovery

class MonoT5Dataset(Dataset):
    def __init__(self, data):
        self.data = data 
  
    def __len__(self):
        return len(self.data)
  
    def __getitem__(self, idx): 
        sample = self.data[idx]
        return {
            'text': sample[0],
            'labels': sample[-1]
        }

def load_train_data(triples_path):
    random.seed(1234)
  
    train_samples = []

    for path in triples_path:
        with open(path, 'r') as f:
            for line in f:
                text, labels = line.split("\t")
                labels = labels.replace('\n', '')
                train_samples.append((text, labels))
        random.shuffle(train_samples)
  
    random.shuffle(train_samples)
    print('len(train_samples): ', len(train_samples))
  
    return train_samples 

def main():

    #args = Namespace(
    #    triples_path = ['/content/drive/MyDrive/NLP_Final_Project/Label_Prediction/scifact_lp_train.txt', 
    #                  '/content/drive/MyDrive/NLP_Final_Project/Label_Prediction/bio_claim_lp_train.txt',
    #                  '/content/drive/MyDrive/NLP_Final_Project/Label_Prediction/bio_query_lp_train.txt'],
    #    output_model_path = '/content/drive/MyDrive/NLP_Final_Project/Label_Prediction/3e-4_6epochs'
    #)

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default='t5-base', type=str, required=False,
                        help="Base model to fine tune.")
    parser.add_argument("--triples_path", default=['/content/drive/MyDrive/NLP_Final_Project/Label_Prediction/scifact_lp_train.txt', 
                                                    '/content/drive/MyDrive/NLP_Final_Project/Label_Prediction/bio_claim_lp_train.txt',
                                                    '/content/drive/MyDrive/NLP_Final_Project/Label_Prediction/bio_query_lp_train.txt'], 
                        type=str, required=True, 
                        help="Triples file paths")
    parser.add_argument("--output_model_path", default='/content/drive/MyDrive/NLP_Final_Project/Label_Prediction/3e-4_6epochs', 
                        type=str, required=True,
                        help="Path for trained model and checkpoints.")
    parser.add_argument("--logging_steps", default=100, type=int, required=False,
                        help="Logging steps parameter.")
    parser.add_argument("--per_device_train_batch_size", default=8, type=int, required=False,
                        help="Per device batch size parameter.")
    parser.add_argument("--gradient_accumulation_steps", default=16, type=int, required=False,
                        help="Gradient accumulation parameter.")
    parser.add_argument("--learning_rate", default=3e-4, type=float, required=False,
                        help="Learning rate parameter.")
    parser.add_argument("--epochs", default=6, type=int, required=False,
                        help="Number of epochs to train")

    device = torch.device('cuda')
    torch.manual_seed(1234)
    args = parser.parse_args()

    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
   
    def smart_batching_collate_text_only(batch):
        texts = [example['text'] for example in batch]
        tokenized = tokenizer(texts, padding=True, truncation='longest_first', return_tensors='pt', max_length=512)
        tokenized['labels'] = tokenizer([example['labels'] for example in batch], return_tensors='pt')['input_ids']

        for name in tokenized: 
            tokenized[name] = tokenized[name].to(device)
    
        return tokenized 

    train_samples = load_train_data(args.triples_path)
    dataset_train = MonoT5Dataset(train_samples)

    train_args = Seq2SeqTrainingArguments(
                output_dir = args.output_model_path,
                do_train = True,
                save_strategy = 'epoch',
                logging_steps = args.logging_steps,
                per_device_train_batch_size = args.per_device_train_batch_size,
                gradient_accumulation_steps = args.gradient_accumulation_steps,
                learning_rate = args.learning_rate, 
                weight_decay = 5e-5,
                num_train_epochs = args.epochs, 
                warmup_steps = 1000,
                adafactor = True, # Use Adafactor optimizer instead of AdamW
                seed = 1234,
                disable_tqdm = False,
                load_best_model_at_end = False,
                predict_with_generate = True,
                dataloader_pin_memory = False,
                report_to = "wandb"
                )

    trainer = Seq2SeqTrainer(
                            model=model,
                            args=train_args,
                            train_dataset=dataset_train,
                            tokenizer=tokenizer,
                            data_collator=smart_batching_collate_text_only
                            )

    trainer.train()

    trainer.save_model(args.output_model_path) 
    trainer.save_state() 

    wandb.finish()


if __name__ == "__main__":
    main()