import warnings
from pathlib import Path
from typing import final

import joblib
import mediapipe as mp
import numpy as np

from mmdemo.base_feature import BaseFeature
from mmdemo.features.gesture.helpers import get_average_hand_pixel, normalize_landmarks, fix_body_id
from mmdemo.interfaces import (
    BodyTrackingInterface,
    CameraCalibrationInterface,
    ColorImageInterface,
    DepthImageInterface,
    FrictionOutputInterface,
    GestureConesInterface,
)
from mmdemo.interfaces.data import Cone, Handedness
from mmdemo.utils.coordinates import CoordinateConversionError, pixel_to_camera_3d
from huggingface_hub import InferenceClient

# - huggingface_hub-0.27.1 #TODO update official yaml


@final
class Friction(BaseFeature[FrictionOutputInterface]):
    """
    Detect friction in group work (client side).

    Input interfaces are TODO add inputs

    Output interface is `FrictionOutputInterface`

    Keyword arguments:
    `model_path` -- the path to the model on hugging face (or None to use the default)
    """
    DEFAULT_MODEL_PATH = "" #TODO is there a default path?

    def __init__(
        self,
        #TODO add inputs
        *,
        model_path: Path | None = None
    ):
        super().__init__() #pass inputs into the base constructor
        if model_path is None:
            self.model_path = self.DEFAULT_MODEL_PATH
        else:
            self.model_path = model_path

        # these are init features to control friction generation quality, context-length etc
        self.generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": True,
            "top_k": top_k,
            "top_p": top_p,
            "num_return_sequences": 1
        }
        self.tags_to_parse = ["friction", "rationale", "t", "b"]
        self.model = None
        self.tokenizer = None
        self.device = None
        self.initialize() # initializes the model and tokenizer loaded from huggingface with additional features
 

    def initialize(self):
        #hugging face setup here
        self.client = InferenceClient()

    def initialize(self):
        """Initialize model and tokenizer"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load base model or LoRA model based on path
        if "checkpoint" in self.model_path:
            self.model = AutoPeftModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            self.model = self.model.merge_and_unload()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.tokenizer.pad_token = "<|reserved_special_token_0|>"
        self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.tokenizer.padding_side = 'right'

    def format_prompt(self, dialogue_history: str) -> str:
        """Format the input prompt for friction generation.
        
     Args:
        dialogue_history (str): Current dialogue transcript with speaker diarization
            
        Example dialogue_history:
        P3: Yeah, maybe let's try it with the smaller one then.
        P1: Just to test like...
        P1: Yeah, I think it's...
        P1: They're just too sturdy and flat, it's like hard to tell.
        P1: We can definitely rule things out but I think uh...
        P1: Too crude of a measurement to...
        P1: Narrow down that like ten gram window.
        P1: Do we want to say eighty?
        P3: I'm confused between eighty and ninety.
        P1: Yeah, it's too balanced.
        P1: Like have another object.
        P3: Uh, there's a scale.
        P1: Are we allowed to use...
        P1: Well we're not even allowed to use the scale. I say we give it our best guess at this point.
        P3: So what should I answer this?
        P3: We have two attempts to do this.
        P1: Ok we have two attempts. Oh well then let's maybe we just try eighty and see if that's right and if it's not then it rules out.
        P3: Yeah we have two attempts. Eighty is good for three ninety. Yeah I'm just clicking on eighty.
        P3: Oh it's right! Eighty, yes!
        P1: Oh we got it, it's right! Eighty, yeah!
        P4: Ok for the final part of the task.
        P4: Read the scenario in the next slide and estimate the weight of the missing second mystery block.
        P4: You will have two attempts to estimate and explain your answer. You can now continue the survey.
        P3: Uh, the factory that creates wooden blocks did not send us a second mystery block.
        P3: Weights so the second block...
        P1: Ok so we literally know nothing about the second block. This sounds like a bayesian probability question, right?
        """



        system_prompt = (
            "You are an expert in collaborative task analysis and personality-driven communication. Think step by step. "
            "Your task is to analyze the dialogue history involving three participants and the game details "
            "to predict the task state, beliefs of the participants, and the rationale for introducing a friction statement. "
            "Finally, generate a nuanced friction statement in a conversational style based on your analysis.\n\n"
            "1. Predict the task-related context and enclose it between the markers `<t>` and `</t>`.\n\n"
            "2. Predict the belief-related context for the participants and enclose it between the markers `<b>` and `</b>`.\n\n"
            "3. Provide a rationale for why a friction statement is needed. This monologue must be enclosed between the "
            "markers `<rationale>` and `</rationale>`. Base your reasoning on evidence from the dialogue, focusing on elements such as:\n"
            "- Incorrect assumptions\n"
            "- False beliefs\n"
            "- Rash decisions\n"
            "- Missing evidence.\n\n"
            "4. Generate the friction statement, ensuring it is enclosed between the markers `<friction>` and `</friction>`. "
            "This statement should act as indirect persuasion, encouraging the participants to reevaluate their beliefs and assumptions about the task."
        )
        
        game_definition = (
            "The game is called 'Game of Weights,' where participants (P1, P2, and P3) determine "
            "the weights of colored blocks. Participants can weigh two blocks at a time and know "
            "the weight of the red block. They must deduce the weights of other blocks."
        )
        
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}. {game_definition}\n\n"
            f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{dialogue_history}\n\n"
            f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            f"### Assistant:"
        )

    def compute_metrics(self, output_ids: torch.Tensor, scores: List[torch.Tensor], prompt_length: int) -> FrictionMetrics:
        """Compute generation metrics"""
        with torch.no_grad():
            logits = torch.stack(scores, dim=0)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            probs = torch.exp(log_probs)
            
            # Get generated token probabilities
            token_ids = output_ids[prompt_length:].cpu()
            probs = probs[:, 0, :].cpu()  # Take first sequence
            token_probs = probs[torch.arange(len(token_ids)), token_ids]
            
            # Calculate metrics
            nll = -torch.sum(torch.log(token_probs)) / len(token_ids)
            predictive_entropy = -torch.sum(probs * log_probs, dim=-1).mean()
            conditional_entropy = -torch.mean(torch.log(token_probs))
            mutual_information = max(predictive_entropy - conditional_entropy, 0.0)
            perplexity = torch.exp(nll)

            return FrictionMetrics(
                nll=nll.item(),
                predictive_entropy=predictive_entropy.item(),
                mutual_information=mutual_information.item(),
                perplexity=perplexity.item(),
                conditional_entropy=conditional_entropy.item()
            )

    def get_output(
        self,
        #inputs to send to model go in this method,
    ):
        #TODO check if any of the input features have been updated
        #if not color.is_new() or not depth.is_new() or not bt.is_new():
            #return None

        #TODO send data to valid hugging face location, await response
        response = self.client.post(json={"inputs": "An astronaut riding a horse on the moon."}, model="stabilityai/stable-diffusion-2-1")
        #response.content

        #TODO return model real reponse


        # first, the only external (demo-side) information that the frition agent needs in the dialouge history at that point where friction needs to be inserted
        # second, the decision to insert friction is made by the planner (I think) but not the f agent
        #third, the format_prompt method will format the input for the friction agent
            #here, along with the dialogue history, a system_prompt and a game_definition is added to properly scaffold the agent. NOTE:system_prompt and game_definition remain constant at any point

        #fourth, now that input is formatted, we'd call the model.generate


    def get_output(self, dialogue_history: str) -> FrictionOutputInterface:
        """Generate friction statement and compute metrics"""
        prompt = self.format_prompt(dialogue_history)
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        
        # we are calling model generate for more freedom on our side to further process the friction agent output (like get metrics etc)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **self.generation_config,
                return_dict_in_generate=True,
                output_scores=True
            )
            
        generated_ids = outputs.sequences[0]
        generated_text = self.tokenizer.decode(generated_ids[inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Parse components and compute metrics
        parsed = self.parse_generation(generated_text)
        metrics = self.compute_metrics(generated_ids, outputs.scores, inputs['input_ids'].shape[1]) # probably not needed
        
        return FrictionOutputInterface(
            friction_statement=parsed.get("friction", ""), #main friction statement
            task_state=parsed.get("t", ""), #state of the task; not important to display 
            belief_state=parsed.get("b", ""), # observed beliefs of participants (about weights) at that point
            rationale=parsed.get("rationale", ""), # this might help make the friction intervention more interpretable
            metrics=metrics, #these will probably not be needed to be exposed in the video but could be effective to have during demo; esp, say confidence of agent
 
        )


        return FrictionOutputInterface(response.content)

