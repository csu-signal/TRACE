import warnings
from pathlib import Path
from typing import final

import joblib
import mediapipe as mp
import numpy as np
import torch
import re
from typing import Dict, List, Optional

from mmdemo.base_feature import BaseFeature
from mmdemo.features.gesture.helpers import get_average_hand_pixel, normalize_landmarks, fix_body_id
from mmdemo.interfaces import (
    BodyTrackingInterface,
    CameraCalibrationInterface,
    ColorImageInterface,
    DepthImageInterface,
    FrictionMetrics,
    FrictionOutputInterface,
    GestureConesInterface,
    TranscriptionInterface,
)

# Hugging Face Libraries
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from mmdemo.interfaces.data import Cone, Handedness
from mmdemo.utils.coordinates import CoordinateConversionError, pixel_to_camera_3d
from huggingface_hub import InferenceClient
from peft import AutoPeftModelForCausalLM, LoraConfig, PeftModel

# - huggingface_hub-0.27.1 #TODO update official yaml
# - accelerate-1.3.0 peft-0.14.0 psutil-6.1.1

# https://huggingface.co/welcome hugging face token
# https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct #TODO update readme

@final
class Friction(BaseFeature[FrictionOutputInterface]):
    """
    Detect friction in group work (client side).

    Input interfaces are `TranscriptionInterface`, ...

    Output interface is `FrictionOutputInterface`

    Keyword arguments:
    `model_path` -- the path to the model on hugging face (or None to use the default)
    """
    # DEFAULT_MODEL_PATH = "Abhijnan/friction_sft_allsamples_weights_instruct"
    DEFAULT_MODEL_PATH = "Abhijnan/friction_agent_sft_merged" 

    def __init__(
        self,
        transcription: BaseFeature[TranscriptionInterface],
        *,
        model_path: Path | None = None
    ):
        super().__init__(transcription) #pass inputs into the base constructor
        if model_path is None:
            self.model_path = self.DEFAULT_MODEL_PATH
        else:
            self.model_path = model_path
        self.transcriptionHistory = ''
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tags_to_parse = ["friction", "rationale", "t", "b"]    

    def initialize(self): ###----> new model initialize function that forces model loading to not use meta parameters in the base model (since lora weights do not contain meta parameters)
    ## assign=True: Ensures that parameters are properly assigned to the correct device, avoiding meta vs. non-meta conflicts.
    ## is_meta=False: Ensures that all parameters are initialized with actual values.
    
        # print("Loading base model...")
        # Add assign=True and is_meta=False to handle meta parameter issues

        # self.model = AutoPeftModelForCausalLM.from_pretrained(
        #     self.model_path,
        #     device_map="auto",
        #     torch_dtype=torch.bfloat16,
        #     trust_remote_code=True,
        #     assign=True,   
        #     is_meta=False   
        # )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
        
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16, # 16 bit precision __
            use_auth_token=True,
            
        )
        
        # print("Merging and unloading adapter weights...", self.model)
        # When merging, also use assign=True
        # self.model = self.model.merge_and_unload(assign=True)
        self.model = self.model.to(self.device)
        print("showing merged lora model", self.model)

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = "<|reserved_special_token_0|>"
        self.tokenizer.padding_side = 'right'

    # def initialize(self):
    #     print("Loading base model...")
    #     self.model = AutoPeftModelForCausalLM.from_pretrained(
    #         self.model_path,
    #         device_map="auto",
    #         torch_dtype=torch.bfloat16,
    #         trust_remote_code=True
    #     )

    #     # Merge and unload adapter weights
    #     print("Merging and unloading adapter weights...",  self.model)
    #     self.model = self.model.merge_and_unload() 
    #     self.model = self.model.to(self.device)  # Move the model to the GPU device
    #     print("showing merged lora model",  self.model)
 
    #     # Load tokenizer
    #     print("Loading tokenizer...")
    #     self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
    #     self.tokenizer.pad_token = "<|reserved_special_token_0|>"
    #     self.tokenizer.padding_side = 'right'

    def parse_generation(self, text: str) -> Dict[str, str]:
        """Parse generated text to extract components"""
        parsed = {tag: [] for tag in self.tags_to_parse}
        
        for tag in self.tags_to_parse:
            pattern = f"<{tag}>(.*?)</{tag}>"
            matches = re.findall(pattern, text, re.DOTALL)
            parsed[tag].extend(matches)
            
        # Handle friction tag specially
        if not parsed["friction"]:
            parsed["friction"] = [self._extract_friction(text)]
            
        return {k: " ".join(v).strip() for k, v in parsed.items()}
    
    def _extract_friction(self, text: str) -> str:
        """Extract friction statement when tags are missing"""
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text.strip())
        if len(sentences) >= 3:
            return f"{sentences[0]} {sentences[-2]} {sentences[-1]}"
        return " ".join(sentences)

    def compute_metrics(self, output_ids: torch.Tensor, scores: List[torch.Tensor], prompt_length: int) -> FrictionMetrics:
        """Compute generation metrics (fixed device handling)"""
        with torch.no_grad():
            # Ensure all tensors are on the same device
            logits = torch.stack(scores, dim=0).to(self.device)
            output_ids = output_ids.to(self.device)
            
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            probs = torch.exp(log_probs)
            
            # Get generated token probabilities
            token_ids = output_ids[prompt_length:]
            probs = probs[:, 0, :]  # Take first sequence
            token_probs = probs[torch.arange(len(token_ids), device=self.device), token_ids]
            
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

    def generate_friction(self, dialogue_history: str, seed: int = 42) -> dict:
        # Format prompt
        torch.manual_seed(seed)
        system_prompt_rm = (
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

        friction_definition_game_definition_prompt_rm = (
        "The game is called 'Game of Weights,' where participants (P1, P2, and P3) determine the weights of colored blocks. "
        "Participants can weigh two blocks at a time and know the weight of the red block. "
        "They must deduce the weights of other blocks. "
        "The dialogue history is provided below:"
        )
 
        # Final formatted prompt for the LLM
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            {system_prompt_rm} 

            {friction_definition_game_definition_prompt_rm}

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            {dialogue_history}

            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            ### Assistant:"""
 
        # Generate response
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)  
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                return_dict_in_generate=True,
                output_scores=True
            )
            generated_ids = outputs.sequences[0]
            generated_ids = generated_ids.to(self.device)
            generated_text = self.tokenizer.decode(
                outputs.sequences[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            scores = outputs.scores  # Already on correct device, don't move to list

            metrics = self.compute_metrics(generated_ids, scores, inputs['input_ids'].shape[1]) # probably not needed
            parsed = self.parse_generation(generated_text)
            #print("\nGenerated text:", generated_text)  # Debug print
            
            # Parse components
            task_state = self._extract_tag(generated_text, "t")
            belief_state = self._extract_tag(generated_text, "b")
            rationale = self._extract_tag(generated_text, "rationale")
            friction = self._extract_tag(generated_text, "friction")
            print("task state:", task_state)
            print("belief state:", belief_state)
            print("rationale:", rationale)
            print("friction:", friction)
            print("metrics", metrics, "\n")

            return FrictionOutputInterface(
                friction_statement=friction,
                task_state=task_state,
                belief_state=belief_state,
                rationale=rationale,
                metrics=metrics,
                raw_generation=generated_text) 
            
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            return None
        
    def _extract_tag(self, text: str, tag: str) -> str:
        import re
        pattern = f"<{tag}>(.*?)</{tag}>"
        matches = re.findall(pattern, text, re.DOTALL)
        return matches[0].strip() if matches else ""
    
    def get_output(self, transcription: TranscriptionInterface):
        if not transcription.is_new():
            return None

        print("\nGetting Friction")
        #self.transcriptionHistory += transcription.speaker_id + ": " + transcription.text + "\n"
        self.transcriptionHistory += "P1: " + transcription.text + "\n"
        print("Transcription History:\n" + self.transcriptionHistory)
        output = self.generate_friction(self.transcriptionHistory)
        return output
    
        # first, the only external (demo-side) information that the frition agent needs in the dialouge history at that point where friction needs to be inserted
        # second, the decision to insert friction is made by the planner (I think) but not the f agent
        #third, the format_prompt method will format the input for the friction agent
            #here, along with the dialogue history, a system_prompt and a game_definition is added to properly scaffold the agent. NOTE:system_prompt and game_definition remain constant at any point
        #fourth, now that input is formatted, we'd call the model.generate

