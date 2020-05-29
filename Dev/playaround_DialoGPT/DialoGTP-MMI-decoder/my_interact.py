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
        self.device = device

        self.num_samples = 10
        self.top_k = 20
        self.mmi_temp = 0.5

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

        self.past_message_list = []

    def append_message(self, message, truncate_length=64):
        if message != '':
            input_token = self.tokenizer.encode(message, return_tensors='pt')
            input_token = torch.cat((input_token, self.end_token), dim=1)
            self.past_message_list.append(input_token)

        if len(self.past_message_list) == 0:
            self.past_message_list.append(self.end_token)

        # truncate
        total_length = 0
        for i, message in enumerate(reversed(self.past_message_list)):
            total_length += message.shape[1]
            if total_length > truncate_length:
                self.past_message_list[:] = self.past_message_list[-i:]

    def generate_message(self, output_token, past):
        out = torch.tensor([[]], dtype=torch.long, device=self.device)

        while True:
            output_token, past = self.forward_model.forward(
                output_token, past=past)
            output_token = output_token[:, -1, :].float()
            indices_to_remove = output_token < torch.topk(
                    output_token, self.top_k)[0][..., -1, None]
            output_token[indices_to_remove] = -float('Inf')
            output_token = torch.multinomial(
                F.softmax(output_token, dim=-1), num_samples=1)

            out = torch.cat((out, output_token), dim=1)

            if output_token.item() == self.end_token.item():
                break

        return out, past

    def score_response(self, output_token, correct_token):
        inputs = torch.cat((output_token, correct_token), dim=1)
        mask = torch.full_like(output_token, -100, dtype=torch.long)
        labels = torch.cat((mask, correct_token), dim=1)

        loss, _, _ = self.reverse_model(inputs, labels=labels)

        return -loss.float()

    def get_response(self, last_message, focus_last_message=True):
        self.append_message(last_message)
        
        total_input = torch.cat(self.past_message_list, dim=1).to(self.device)
        if focus_last_message:
            total_input_reversed = self.past_message_list[-1]
        else:
            total_input_reversed = torch.cat(
                list(reversed(self.past_message_list)), dim=1)

        past = None
        if total_input.shape[1] > 1:
            _, past = self.forward_model(total_input[:, :-1])

        results = []
        for i in range(self.num_samples):
            result = self.generate_message(total_input[:, -1:], past)
            score = self.score_response(
                result[0].to(self.device), 
                total_input_reversed.to(self.device)
            )
            results.append(result + (score,))

        scores = torch.stack([x[2] for x in results], dim=0)
        winner = torch.multinomial(
            F.softmax(scores / self.mmi_temp, dim=0), num_samples=1).item()
        # winner = torch.argmax(scores, dim=0)

        out = results[winner][0]

        generated_message = self.tokenizer.decode(
            out.tolist()[0], skip_special_tokens=True)

        print(self.name + ' >> ' + generated_message)

        return generated_message

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

    last_message = ''
    while True:
        last_message = agent1.get_response(last_message)
        if last_message == 'quit':
            break
        last_message = agent2.get_response(last_message)
        if last_message == 'quit':
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Have a conversation with different agents.')
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