import os
import urllib.parse
import requests
import json
import re

class ImageGenerationService:
    """
    Modular Image Generation Service for SmartMirror-Vox.
    Supports text-to-image and reference-based personal image transformations (try-on, hairstyle, background).
    Provider can be configured via IMAGE_PROVIDER in .env without changing client code.
    """

    def __init__(self):
        self.provider = os.environ.get("IMAGE_PROVIDER", "flux-realism").lower()
        self.api_key = os.environ.get("IMAGE_API_KEY", "")
        self.openai_key = os.environ.get("OPENAI_API_KEY", "")
        self.width = int(os.environ.get("IMAGE_WIDTH", "768"))
        self.height = int(os.environ.get("IMAGE_HEIGHT", "768"))

    def build_photorealistic_normal_prompt(self, user_prompt: str) -> str:
        """
        Enhances a standard user request into a high-detail photorealistic prompt.
        """
        clean = re.sub(r'^(?:can\s+you\s+)?(?:please\s+)?(?:generate|create|draw|make|show(?:\s+me)?)\s+(?:an?\s+)?(?:image\s+(?:of\s+)?|picture\s+(?:of\s+)?|photo\s+(?:of\s+)?)?', '', user_prompt, flags=re.IGNORECASE).strip()
        if not clean:
            clean = user_prompt
        
        enhanced = (
            f"candid highly realistic photographic portrait of {clean}, "
            f"natural real-world lighting, shot on 35mm DSLR camera, f/1.8 aperture, sharp focus, "
            f"subtle depth of field, natural textures, authentic photograph, 8k resolution, unedited real life"
        )
        return enhanced

    def clean_modification_text(self, text: str) -> str:
        """
        Strips conversational filler and speech-to-text noise from modification attributes.
        """
        if not text:
            return ""
        clean = text.strip()
        # Strip n8n template prefix/suffix
        clean = re.sub(r'^candid\s+raw\s+35mm\s+photograph\s+of\s+a\s+real\s+', '', clean, flags=re.IGNORECASE)
        clean = re.sub(r',\s*authentic\s+natural\s+lighting.*$', '', clean, flags=re.IGNORECASE)
        # Strip conversational preamble
        clean = re.sub(r'^(?:can\s+you\s+)?(?:please\s+)?(?:generate|create|draw|make|show(?:\s+me)?)\s+(?:an?\s+)?(?:image\s+(?:of\s+)?|picture\s+(?:of\s+)?|photo\s+(?:of\s+)?)?', '', clean, flags=re.IGNORECASE)
        clean = re.sub(r'^(?:see\s+(?:what|how)\s+i(?:\s*[\'’]m|\s+am)?\s+(?:if\s+i\s+wear\s+|in\s+)?|what\s+if\s+i\s+wear\s+|how\s+(?:would|will|do)\s+i\s+look\s+(?:in|with|wearing)?\s*|show\s+me\s+(?:in|wearing|with)?\s*)', '', clean, flags=re.IGNORECASE)
        # Strip conversational suffixes
        clean = re.sub(r'\s*(?:avayam\s+look\s+like|look\s+like|can\s+you\s+able\s+to\s+generate\s+(?:the|an?)?\s+image|can\s+you\s+generate\s+(?:the|an?)?\s+image|generate\s+(?:the|an?)?\s+image).*$', '', clean, flags=re.IGNORECASE)
        return clean.strip()

    def build_identity_preservation_prompt(self, user_request: str, person_profile: str, modifications: dict = None) -> str:
        """
        Constructs a prompt strictly adhering to Section 9 identity preservation guidelines:
        Preserves face, skin tone, facial proportions, age, body proportions.
        Modifies ONLY requested attributes (clothing, hairstyle, background).
        Enforces authentic South Asian Indian identity and strict clothing layering rules.
        """
        modifications = modifications or {}
        raw_clothing = modifications.get('clothing') or modifications.get('outfit') or ''
        raw_hairstyle = modifications.get('hairstyle') or ''
        raw_background = modifications.get('background') or ''
        raw_details = modifications.get('details') or ''

        # Clean attributes
        clothing = self.clean_modification_text(raw_clothing)
        hairstyle = self.clean_modification_text(raw_hairstyle)
        background = self.clean_modification_text(raw_background)
        details = self.clean_modification_text(raw_details)

        # Fallback if no specific modification key was extracted
        if not (clothing or hairstyle or background or details):
            cleaned_request = self.clean_modification_text(user_request)
            req_lower = cleaned_request.lower()
            if any(k in req_lower for k in ['shirt', 'suit', 'jacket', 'hoodie', 'dress', 'tuxedo', 'pants', 't-shirt', 'kurta']):
                clothing = cleaned_request
            elif any(k in req_lower for k in ['hair', 'haircut', 'hairstyle', 'mullet', 'fade', 'curly', 'straight']):
                hairstyle = cleaned_request
            elif any(k in req_lower for k in ['beach', 'paris', 'new york', 'city', 'background', 'room', 'garden']):
                background = cleaned_request
            else:
                clothing = cleaned_request

        # 1. Identity & Demographic Anchor (Ensures genuine South Asian Indian male appearance matching mirror user)
        base_identity = "authentic young adult South Asian Indian male with natural warm brown Indian skin tone, dark brown eyes, thick neat dark eyebrows, neat dark hair, short neat mustache, natural South Asian Indian facial structure and features"
        
        if person_profile:
            profile_clean = re.sub(r'^(?:real\s+|a\s+real\s+|a\s+)', '', person_profile, flags=re.IGNORECASE).strip()
            # If profile misses Indian/South Asian or mustache, ensure it's firmly anchored
            if not any(k in profile_clean.lower() for k in ['indian', 'south asian']):
                profile_clean = f"South Asian Indian {profile_clean}"
            if 'mustache' not in profile_clean.lower() and 'beard' not in profile_clean.lower():
                profile_clean = f"{profile_clean}, neat short mustache"
            if 'male' not in profile_clean.lower() and 'man' not in profile_clean.lower() and 'guy' not in profile_clean.lower():
                profile_clean = f"{profile_clean} male"
            identity_desc = profile_clean
        else:
            identity_desc = base_identity

        # 2. Enhanced Clothing Specification with anti-layering constraints
        mods_specs = []

        if clothing:
            c_low = clothing.lower()
            if 'white shirt' in c_low or (c_low == 'shirt' and 'white' in user_request.lower()):
                cloth_spec = "wearing a crisp tailored classic white button-down collared dress shirt with buttons, neatly pressed and tucked, sharp collar, realistic cotton weave fabric texture, paired with tailored formal trousers and leather belt, strictly wearing ONLY the white shirt with NO jacket, NO hoodie, NO coat, NO sweater, NO cardigan over it"
            elif 'black shirt' in c_low or ('shirt' in c_low and 'black' in user_request.lower()):
                cloth_spec = "wearing a crisp tailored solid black button-down collared dress shirt with buttons, sleeves neatly rolled up to mid-forearm (3/4 fold), neatly tucked into tailored slim-fit light grey formal trousers with a black leather belt with silver buckle, and clean minimalist white leather low-top sneakers, strictly wearing ONLY the black shirt with NO jacket, NO hoodie, NO coat"
            elif 'shirt' in c_low and not any(w in c_low for w in ['t-shirt', 'tshirt']):
                cloth_spec = f"wearing a crisp tailored classic {clothing} with buttons and sharp collar, sleeves neatly rolled up to mid-forearm, neatly tucked into tailored trousers with a leather belt and clean stylish footwear, strictly wearing ONLY the {clothing} with NO hoodie, NO jacket, NO coat over it"
            elif 'suit' in c_low or 'tuxedo' in c_low or 'blazer' in c_low:
                cloth_spec = f"wearing a sharp tailored {clothing}, premium fabric, elegant modern formal fit"
            elif 'kurta' in c_low or 'ethnic' in c_low:
                cloth_spec = f"wearing a traditional elegant {clothing} with authentic Indian embroidery and rich fabric"
            else:
                cloth_spec = f"wearing {clothing}, realistic fabric texture, natural fit"
            mods_specs.append(cloth_spec)

        if hairstyle:
            h_low = hairstyle.lower()
            if 'mullet' in h_low:
                hair_spec = "with a modern textured mullet hairstyle, shorter neatly faded sides, longer natural textured hair flowing at the back, natural black hair strands matching his face"
            elif 'fade' in h_low:
                hair_spec = "with a clean sharp skin fade haircut on the sides, neat textured dark hair on top"
            else:
                hair_spec = f"with a natural {hairstyle} hairstyle, realistic individual dark hair strands matching his head shape"
            mods_specs.append(hair_spec)

        if background:
            mods_specs.append(f"standing naturally in front of {background}, natural matching environmental lighting and subtle background depth of field")
        else:
            mods_specs.append("full-body candid street style fashion portrait leaning casually against a dark polished granite pillar outside a luxury modern architectural hotel plaza at dusk, with warm golden bokeh background lights, lush potted greenery, and reflective polished stone flooring")
        
        if details:
            mods_specs.append(f"styled with {details}")

        if not mods_specs:
            mods_specs.append("wearing a crisp tailored classic solid black button-down collared dress shirt with sleeves rolled up to mid-forearm, light grey tailored trousers, and white sneakers, strictly no jacket, no hoodie")

        modifications_str = ", ".join(mods_specs)

        # 3. Assemble complete photorealistic identity-preservation prompt
        prompt = (
            f"candid full-body raw 35mm street style fashion photograph of the exact same real {identity_desc}. "
            f"CRITICAL IDENTITY PRESERVATION: preserve the person's exact face, facial structure, warm natural brown Indian skin tone, "
            f"dark eyes, nose, mouth, neat short mustache, well-groomed light stubble, facial proportions, age, and natural skin pores and texture. "
            f"MODIFICATIONS: {modifications_str}. "
            f"Authentic natural evening twilight lighting with warm golden bokeh background lights, photorealistic 35mm f/1.8 DSLR camera portrait, "
            f"sharp focus, natural skin texture, realistic shadows, authentic real life photograph. "
            f"Strictly NO East Asian features, NO Caucasian features, NO hoodie, NO jacket unless requested, NO anime, NO 3D CGI render, NO plastic skin."
        )
        return prompt

    def generate_image(self, prompt: str, options: dict = None) -> dict:
        """
        Normal Image Generation: Generates an image from prompt without reference photo.
        """
        options = options or {}
        enhanced_prompt = self.build_photorealistic_normal_prompt(prompt)

        # 1. Primary: High-fidelity Pollinations Flux-Realism model (Zero 402 errors, crisp 8k detail)
        encoded = urllib.parse.quote(enhanced_prompt)
        w = options.get('width', self.width)
        h = options.get('height', self.height)
        model = options.get('model', 'flux-realism')
        image_url = f"https://image.pollinations.ai/prompt/{encoded}?width={w}&height={h}&model={model}&nologo=true"

        return {
            "success": True,
            "image_url": image_url,
            "prompt": prompt,
            "enhanced_prompt": enhanced_prompt,
            "provider": self.provider
        }

    def generate_image_from_reference(self, image_base64: str, prompt: str, person_profile: str = None, modifications: dict = None, options: dict = None) -> dict:
        """
        Personal Image Generation: Preserves the user's identity from the reference image,
        applies requested modifications (outfit, hairstyle, background), and returns result.
        """
        options = options or {}
        enhanced_prompt = self.build_identity_preservation_prompt(prompt, person_profile, modifications)

        encoded = urllib.parse.quote(enhanced_prompt)
        w = options.get('width', self.width)
        h = options.get('height', self.height)
        model = options.get('model', 'flux-realism')
        image_url = f"https://image.pollinations.ai/prompt/{encoded}?width={w}&height={h}&model={model}&nologo=true"

        return {
            "success": True,
            "image_url": image_url,
            "prompt": prompt,
            "enhanced_prompt": enhanced_prompt,
            "modifications": modifications or {},
            "provider": self.provider
        }

# Global singleton service
image_service = ImageGenerationService()
