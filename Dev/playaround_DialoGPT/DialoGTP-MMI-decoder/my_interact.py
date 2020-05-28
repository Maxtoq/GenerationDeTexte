import os
import argparse
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config


DIALOGPT_MODELS = 'medium'


class HumanAgent:
    def __init__(self, name):
        self.name = name

    def get_response(self, last_message):
        response = input(self.name + ' >> ')
        return response

class DialoGPTAgent:
    def __init__(self, name, device, half_p=False):
        self.name = name

        self.tokenizer = GPT2Tokenizer(
            os.path.join(DIALOGPT_MODELS, 'vocab.json'), 
            os.path.join(DIALOGPT_MODELS, 'merges.txt')
        )

        cfg = GPT2Config.from_json_file(
            os.path.join(DIALOGPT_MODELS, 'config.json'))
        self.forward_model = self.prepare_model(
            cfg,
            os.path.join(DIALOGPT_MODELS, 'medium_ft.pkl'),
            device,
            half_p
        )
        self.reverse_model = self.prepare_model(
            cfg,
            os.path.join(DIALOGPT_MODELS, 'small_reverse.pkl'),
            device,
            half_p
        )

        self.end_token = torch.tensor([[50256]], dtype=torch.long)

    @staticmethod
    def prepare_model(config, weights_path, device, half_p):
        model =  GPT2LMHeadModel(config)
        # load and fix weights
        weights = torch.load(weights_path)
        weights["lm_head.weight"] = weights["lm_head.decoder.weight"]
        weights.pop("lm_head.decoder.weight", None)
        model.load_state_dict(weights)
        if half_p:
            model.half()
        model.to(device)
        model.eval()
        return model


def get_agent(agent_type, number, device, half_p):
    name = f'Agent #{number} ({agent_type})'
    if agent_type == 'human':
        return HumanAgent(name)
    elif agent_type == 'dialogpt':
        return DialoGPTAgent(name, device, half_p)

def main(args):
    torch.set_grad_enabled(False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    agent1 = get_agent(args.agent1, 1, device, args.half)
    agent2 = get_agent(args.agent2, 2, device, args.half)

    last_message = None
    while True:
        last_message = agent1.get_response(last_message)
        last_message = agent2.get_response(last_message)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Have a conversation with different agents.')
    parser.add_argument('--half', 
        help='enable half precision (FP16) for model computation.',
        action='store_true'
    )
    parser.add_argument('-a1', '--agent1', 
        help='-a1 <agent_name> : name of the first conversational agent: \
              \'human\' or \'dialogpt\' (default is \'human\')',
        type=str,
        default='human'
    )
    parser.add_argument('-a2', '--agent2', 
        help='-a2 <agent_name> : name of the second conversational agent: \
              \'human\' or \'dialogpt\' (default is \'dialogpt\')',
        type=str,
        default='dialogpt'
    )
    args = parser.parse_args()

    main(args)