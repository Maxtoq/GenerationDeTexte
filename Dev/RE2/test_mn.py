import os
import sys
from src.model import Model
from src.interface import Interface
from src.utils.loader import load_data


class MemoryNetwork:
    def __init__(self, init_mem=[]):
        self.memory = init_mem

    def store(self, new_sentence):
        self.memory.append(new_sentence)

class NLIModel:
    def __init__(self, model_path):
        self.model, self.checkpoint = Model.load(model_path)
        self.interface = Interface(self.checkpoint['args'])

    def score_sentence(self, sentence, memory):
        if len(memory) == 0:
            return 1.0
            
        batch = []
        for past_sentence in memory:
            sample = {
                'text1': past_sentence,
                'text2': sentence
            }
            processed = self.interface.process_sample(sample, with_target=False)

            batch.append(processed)

        batch = self.interface.make_batch(batch, with_target=False)

        predictions = self.model.predict(batch)
        return predictions


def main():
    argv = sys.argv

    init_memory = [
        'I love basketball.',
        'I work at the mall.',
        'The professor is very angry.',
        'She was crying earlier.'
    ]

    mn = MemoryNetwork(init_memory)
    model = NLIModel(argv[1])

    while True:
        usr = input('usr >> ')

        if usr == 'quit':
            break

        print('Entailment with sentence > \"' + usr + '\":')
        predictions = model.score_sentence(usr, mn.memory)
        
        if type(predictions) is float:
            continue

        for sentence, score in zip(mn.memory, predictions):
            print(sentence, score)

        mn.store(usr)
        print('New MN:')
        print(mn.memory)

if __name__ == '__main__':
    main()
