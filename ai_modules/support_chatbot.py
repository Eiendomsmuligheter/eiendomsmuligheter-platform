from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
import json
import numpy as np
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

logger = logging.getLogger(__name__)

class SupportChatbot:
    def __init__(self):
        self.model_name = "norwegian-nlp/norwegian-gpt3"  # Placeholder
        self.knowledge_base = self._load_knowledge_base()
        self.conversation_history = []
        self.tokenizer = None
        self.model = None
        
    def _load_knowledge_base(self) -> Dict[str, Any]:
        """Last inn kunnskapsbase for chatboten"""
        return {
            "regulations": {
                "TEK17": {
                    "sections": {},
                    "common_questions": []
                },
                "Plan_og_bygningsloven": {
                    "sections": {},
                    "common_questions": []
                }
            },
            "application_process": {
                "steps": [],
                "requirements": [],
                "common_issues": []
            },
            "technical_requirements": {
                "fire_safety": [],
                "ventilation": [],
                "accessibility": []
            }
        }
        
    def _initialize_model(self):
        """Initialiser språkmodellen"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        except Exception as e:
            logger.error(f"Feil ved lasting av språkmodell: {str(e)}")
            
    def process_question(self, question: str) -> str:
        """Prosesser spørsmål og generer svar"""
        try:
            # Legg til spørsmål i historikk
            self.conversation_history.append({"role": "user", "content": question})
            
            # Analyser spørsmålet
            intent = self._analyze_intent(question)
            context = self._get_relevant_context(intent)
            
            # Generer svar
            response = self._generate_response(question, context)
            
            # Legg til svar i historikk
            self.conversation_history.append({"role": "assistant", "content": response})
            
            return response
            
        except Exception as e:
            logger.error(f"Feil ved prosessering av spørsmål: {str(e)}")
            return "Beklager, jeg kunne ikke prosessere spørsmålet ditt. Prøv å omformulere det."
            
    def _analyze_intent(self, question: str) -> Dict[str, float]:
        """Analyser intensjonen med spørsmålet"""
        if not self.model:
            self._initialize_model()
            
        # Klassifiser spørsmålstype
        keywords = {
            "technical_requirements": [
                "krav", "teknisk", "ventilasjon", "brann", "sikkerhet",
                "isolasjon", "størrelse", "areal", "høyde", "vindu"
            ],
            "regulations": [
                "lov", "forskrift", "tek17", "regulering", "plan",
                "byggesak", "tillatelse", "søknad", "kommune"
            ],
            "process": [
                "hvordan", "prosess", "søke", "steg", "fremgangsmåte",
                "tidslinje", "kostnad", "gebyr", "behandling"
            ],
            "economic": [
                "pris", "kostnad", "inntekt", "lån", "finansiering",
                "investering", "roi", "avkastning", "utgift"
            ]
        }
        
        scores = {}
        question_lower = question.lower()
        
        for category, words in keywords.items():
            score = sum(1 for word in words if word in question_lower)
            confidence = min(score / len(words) + 0.3, 1.0)
            scores[category] = confidence
            
        # Finn kategorien med høyest score
        max_category = max(scores.items(), key=lambda x: x[1])
        
        return {
            "category": max_category[0],
            "confidence": max_category[1],
            "all_scores": scores
        }
        
    def _get_relevant_context(self, intent: Dict[str, float]) -> Dict[str, Any]:
        """Hent relevant kontekst basert på intensjon"""
        category = intent["category"]
        context = {"relevant_info": [], "references": []}
        
        if category == "technical_requirements":
            context["relevant_info"].extend([
                "Minimumskrav til takhøyde er 2.4 meter i oppholdsrom",
                "Alle rom må ha tilstrekkelig ventilasjon",
                "Brannsikring krever røykvarslere og rømningsveier",
                "Vinduer må utgjøre minimum 10% av gulvarealet"
            ])
            context["references"].extend([
                "TEK17 § 12-7. Krav til utforming",
                "TEK17 § 13-1. Ventilasjon",
                "TEK17 § 11-2. Brannsikring"
            ])
            
        elif category == "regulations":
            context["relevant_info"].extend([
                "Bruksendring krever søknad til kommunen",
                "Nabovarsel må sendes minst 14 dager før søknad",
                "Tiltaket må være i tråd med reguleringsplan",
                "Ansvarlig søker må være godkjent av kommunen"
            ])
            context["references"].extend([
                "Plan- og bygningsloven § 20-1",
                "Byggesaksforskriften § 5-2",
                "Kommunens arealplan"
            ])
            
        elif category == "process":
            context["relevant_info"].extend([
                "Søknadsprosessen tar normalt 3-12 uker",
                "Komplett søknad må inkludere tegninger og dokumentasjon",
                "Behandlingsgebyr varierer mellom kommuner",
                "Midlertidig brukstillatelse kan søkes"
            ])
            context["references"].extend([
                "Byggesaksforskriften § 7-1",
                "Kommunens gebyrregulativ",
                "Veileder for søknadsprosessen"
            ])
            
        elif category == "economic":
            context["relevant_info"].extend([
                "Typiske oppgraderingskostnader er 5000-15000 kr/m²",
                "ROI varierer mellom 8-15% årlig",
                "Utleiepriser varierer med beliggenhet og standard",
                "Skattemessige fordeler ved utleie"
            ])
            context["references"].extend([
                "Skatteloven § 7-2",
                "Husleieloven",
                "Lokale markedsanalyser"
            ])
            
        # Legg til historisk kontekst fra samtalen
        if len(self.conversation_history) > 2:
            context["conversation_context"] = self.conversation_history[-4:]
            
        return context
        
    def _generate_response(self, question: str, context: Dict[str, Any]) -> str:
        """Generer svar basert på spørsmål og kontekst"""
        if not self.model:
            self._initialize_model()
            
        try:
            # Forbered prompt med kontekst
            prompt = self._prepare_prompt(question, context)
            
            # Generer svar med språkmodell
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            outputs = self.model.generate(
                inputs,
                max_length=200,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Formater og valider svaret
            formatted_response = self._format_response(response, context)
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Feil ved generering av svar: {str(e)}")
            
            # Fallback: Returner relevant informasjon direkte fra kontekst
            relevant_info = context.get("relevant_info", [])
            if relevant_info:
                return f"Basert på ditt spørsmål, her er relevant informasjon:\n\n" + \
                       "\n".join(f"- {info}" for info in relevant_info[:2])
            else:
                return "Beklager, jeg kunne ikke generere et presist svar. " + \
                       "Vennligst kontakt kundeservice for mer informasjon."
                       
    def _prepare_prompt(self, question: str, context: Dict[str, Any]) -> str:
        """Forbered prompt for språkmodellen"""
        # Start med spørsmålet
        prompt = f"Spørsmål: {question}\n\n"
        
        # Legg til relevant kontekst
        if context.get("relevant_info"):
            prompt += "Relevant informasjon:\n"
            prompt += "\n".join(f"- {info}" for info in context["relevant_info"])
            prompt += "\n\n"
            
        # Legg til tidligere samtalehistorikk hvis relevant
        if context.get("conversation_context"):
            prompt += "Tidligere i samtalen:\n"
            for msg in context["conversation_context"]:
                role = "Bruker" if msg["role"] == "user" else "Assistent"
                prompt += f"{role}: {msg['content']}\n"
            prompt += "\n"
            
        # Legg til instruksjoner for svarformat
        prompt += "Gi et klart og konsist svar på norsk som:\n"
        prompt += "1. Adresserer spørsmålet direkte\n"
        prompt += "2. Inkluderer relevant teknisk informasjon\n"
        prompt += "3. Refererer til gjeldende forskrifter\n"
        prompt += "4. Gir praktiske anbefalinger\n\n"
        
        prompt += "Svar:"
        
        return prompt
        
    def _format_response(self, response: str, context: Dict[str, Any]) -> str:
        """Formater og forbedre svaret"""
        lines = response.split('\n')
        formatted_lines = []
        
        # Fjern tomme linjer og formater teksten
        for line in lines:
            line = line.strip()
            if line and not line.startswith("Svar:"):
                formatted_lines.append(line)
                
        response = "\n".join(formatted_lines)
        
        # Legg til referanser hvis tilgjengelig
        if context.get("references"):
            response += "\n\nReferanser:\n"
            response += "\n".join(f"- {ref}" for ref in context["references"][:2])
            
        return response
        
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Generer sammendrag av samtalen"""
        if not self.conversation_history:
            return {
                "total_exchanges": 0,
                "main_topics": [],
                "resolution_status": "no_conversation",
                "duration": 0,
                "sentiment": "neutral"
            }
            
        try:
            analysis = {
                "total_exchanges": len(self.conversation_history) // 2,
                "main_topics": self._analyze_conversation_topics(),
                "resolution_status": self._check_resolution_status(),
                "duration": self._calculate_conversation_duration(),
                "sentiment": self._analyze_sentiment(),
                "key_questions": self._extract_key_questions(),
                "topics_by_frequency": self._analyze_topic_frequency(),
                "technical_terms_used": self._extract_technical_terms(),
                "action_items": self._extract_action_items()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Feil ved generering av samtalesammendrag: {str(e)}")
            return {
                "total_exchanges": len(self.conversation_history) // 2,
                "error": "Kunne ikke generere fullstendig analyse"
            }
        
    def _analyze_conversation_topics(self) -> List[Dict[str, Any]]:
        """Analyser hovedtemaer i samtalen"""
        topics = []
        current_topic = None
        topic_messages = []
        
        for message in self.conversation_history:
            if message["role"] == "user":
                # Analyser intensjon for hver brukermelding
                intent = self._analyze_intent(message["content"])
                
                # Hvis ny kategori eller første melding
                if not current_topic or intent["category"] != current_topic["category"]:
                    # Lagre forrige tema hvis det finnes
                    if current_topic:
                        topics.append({
                            **current_topic,
                            "message_count": len(topic_messages),
                            "duration": self._calculate_topic_duration(topic_messages),
                            "key_questions": self._extract_key_questions_from_messages(topic_messages)
                        })
                    
                    # Start nytt tema
                    current_topic = {
                        "category": intent["category"],
                        "confidence": intent["confidence"],
                        "start_time": message.get("timestamp", datetime.now().isoformat())
                    }
                    topic_messages = [message]
                else:
                    topic_messages.append(message)
        
        # Legg til siste tema
        if current_topic:
            topics.append({
                **current_topic,
                "message_count": len(topic_messages),
                "duration": self._calculate_topic_duration(topic_messages),
                "key_questions": self._extract_key_questions_from_messages(topic_messages)
            })
        
        return topics
        
    def _check_resolution_status(self) -> str:
        """Sjekk om brukerens spørsmål ble besvart"""
        if not self.conversation_history:
            return "no_conversation"
            
        # Analyser siste utveksling
        last_exchange = self.conversation_history[-2:] if len(self.conversation_history) >= 2 else []
        
        if not last_exchange:
            return "incomplete"
            
        # Sjekk etter takkemeldinger eller bekreftelser fra bruker
        positive_indicators = [
            "takk", "forstår", "skjønner", "bra", "flott", "perfekt",
            "hjelpsom", "nyttig", "det hjelper"
        ]
        
        negative_indicators = [
            "fortsatt ikke", "forstår ikke", "uklart", "forvirret",
            "hjelper ikke", "ikke svart", "samme spørsmål"
        ]
        
        # Hvis siste melding er fra bruker
        if last_exchange[-1]["role"] == "user":
            message = last_exchange[-1]["content"].lower()
            
            if any(indicator in message for indicator in positive_indicators):
                return "resolved"
            elif any(indicator in message for indicator in negative_indicators):
                return "unresolved"
            else:
                return "in_progress"
                
        # Hvis siste melding er fra assistent
        elif len(last_exchange) == 2:
            user_message = last_exchange[0]["content"].lower()
            assistant_message = last_exchange[1]["content"].lower()
            
            # Sjekk om assistenten ga et substansielt svar
            if len(assistant_message) > 100 and "beklager" not in assistant_message.lower():
                return "awaiting_confirmation"
            else:
                return "needs_followup"
                
        return "unknown"
        
    def _calculate_conversation_duration(self) -> int:
        """Beregn samtalens varighet i sekunder"""
        if not self.conversation_history:
            return 0
            
        try:
            start_time = datetime.fromisoformat(self.conversation_history[0].get("timestamp", ""))
            end_time = datetime.fromisoformat(self.conversation_history[-1].get("timestamp", ""))
            return (end_time - start_time).total_seconds()
        except:
            return 0
            
    def _analyze_sentiment(self) -> str:
        """Analyser sentiment i samtalen"""
        if not self.conversation_history:
            return "neutral"
            
        # Sentiment-indikatorer
        positive_words = set([
            "takk", "bra", "flott", "perfekt", "hjelpsom", "nyttig",
            "fornøyd", "utmerket", "strålende", "super"
        ])
        
        negative_words = set([
            "ikke", "dårlig", "vanskelig", "problem", "uklart", "forvirret",
            "misfornøyd", "irriterende", "frustrerende"
        ])
        
        # Tell forekomster
        positive_count = 0
        negative_count = 0
        
        for message in self.conversation_history:
            if message["role"] == "user":
                words = message["content"].lower().split()
                positive_count += sum(1 for word in words if word in positive_words)
                negative_count += sum(1 for word in words if word in negative_words)
                
        # Bestem sentiment
        if positive_count > negative_count * 1.5:
            return "positive"
        elif negative_count > positive_count * 1.5:
            return "negative"
        else:
            return "neutral"
            
    def _extract_key_questions(self) -> List[str]:
        """Hent ut nøkkelspørsmål fra samtalen"""
        key_questions = []
        
        for message in self.conversation_history:
            if message["role"] == "user":
                content = message["content"]
                # Identifiser spørsmål basert på spørreord eller spørsmålstegn
                if any(word in content.lower() for word in ["hvordan", "hva", "hvor", "når", "hvilken", "kan"]) or "?" in content:
                    # Fjern unødvendig whitespace og formatering
                    question = " ".join(content.split())
                    if len(question) > 10:  # Ignorer for korte spørsmål
                        key_questions.append(question)
                        
        return key_questions
        
    def _analyze_topic_frequency(self) -> Dict[str, int]:
        """Analyser frekvensen av ulike temaer"""
        topic_count = {}
        
        for message in self.conversation_history:
            if message["role"] == "user":
                intent = self._analyze_intent(message["content"])
                category = intent["category"]
                topic_count[category] = topic_count.get(category, 0) + 1
                
        return dict(sorted(topic_count.items(), key=lambda x: x[1], reverse=True))
        
    def _extract_technical_terms(self) -> List[Dict[str, Any]]:
        """Hent ut tekniske termer brukt i samtalen"""
        technical_terms = set()
        term_contexts = {}
        
        # Liste over tekniske termer å se etter
        terms_to_track = {
            "TEK17": "byggteknisk forskrift",
            "bruksendring": "endring av bygningens bruk",
            "reguleringsplan": "plan for arealbruk",
            "branncelle": "brannsikker avdeling",
            "rømningsvei": "vei for evakuering",
            "universell utforming": "tilgjengelighet for alle",
            "etasjeskiller": "konstruksjon mellom etasjer",
            "luftemengde": "ventilasjonskrav",
            "lydklasse": "krav til lydisolasjon"
        }
        
        for message in self.conversation_history:
            content = message["content"].lower()
            for term, description in terms_to_track.items():
                if term.lower() in content:
                    technical_terms.add(term)
                    # Lagre kontekst for termen
                    if term not in term_contexts:
                        term_contexts[term] = {
                            "description": description,
                            "frequency": 1,
                            "first_mentioned_by": message["role"]
                        }
                    else:
                        term_contexts[term]["frequency"] += 1
                        
        return [
            {
                "term": term,
                "description": term_contexts[term]["description"],
                "frequency": term_contexts[term]["frequency"],
                "first_mentioned_by": term_contexts[term]["first_mentioned_by"]
            }
            for term in technical_terms
        ]
        
    def _extract_action_items(self) -> List[Dict[str, Any]]:
        """Hent ut handlingspunkter fra samtalen"""
        action_items = []
        
        # Nøkkelord som indikerer handlingspunkter
        action_indicators = [
            "må", "skal", "bør", "trenger å", "husk å", "ikke glem",
            "viktig å", "nødvendig å", "anbefaler å"
        ]
        
        for idx, message in enumerate(self.conversation_history):
            if message["role"] == "assistant":
                content = message["content"]
                
                # Finn setninger med handlingsindikatorer
                sentences = content.split(". ")
                for sentence in sentences:
                    if any(indicator in sentence.lower() for indicator in action_indicators):
                        action_items.append({
                            "action": sentence.strip(),
                            "context": self._get_conversation_context(idx),
                            "priority": self._determine_action_priority(sentence),
                            "category": self._categorize_action(sentence)
                        })
                        
        return action_items
        
    def _get_conversation_context(self, message_idx: int, context_window: int = 2) -> str:
        """Hent kontekst rundt en melding"""
        start_idx = max(0, message_idx - context_window)
        end_idx = min(len(self.conversation_history), message_idx + context_window + 1)
        
        context_messages = []
        for i in range(start_idx, end_idx):
            msg = self.conversation_history[i]
            role = "Bruker" if msg["role"] == "user" else "Assistent"
            context_messages.append(f"{role}: {msg['content']}")
            
        return "\n".join(context_messages)
        
    def _determine_action_priority(self, action: str) -> str:
        """Bestem prioritet for et handlingspunkt"""
        high_priority_indicators = ["må", "kritisk", "umiddelbart", "snarest", "ikke utsett"]
        medium_priority_indicators = ["bør", "anbefaler", "viktig"]
        
        action_lower = action.lower()
        
        if any(indicator in action_lower for indicator in high_priority_indicators):
            return "høy"
        elif any(indicator in action_lower for indicator in medium_priority_indicators):
            return "medium"
        else:
            return "lav"
            
    def _categorize_action(self, action: str) -> str:
        """Kategoriser et handlingspunkt"""
        categories = {
            "dokumentasjon": ["søknad", "skjema", "dokumenter", "tegninger"],
            "teknisk": ["brannsikring", "ventilasjon", "konstruksjon"],
            "juridisk": ["forskrift", "regulering", "tillatelse"],
            "økonomisk": ["kostnad", "betaling", "gebyr", "finansiering"],
            "praktisk": ["måle", "installere", "kontakte", "sjekke"]
        }
        
        action_lower = action.lower()
        
        for category, keywords in categories.items():
            if any(keyword in action_lower for keyword in keywords):
                return category
                
        return "annet"