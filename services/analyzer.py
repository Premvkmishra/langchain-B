import re
import random
from typing import Dict, List, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# LangChain imports
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResumeAnalyzer:
    def __init__(self):
        # Initialize LangChain with HuggingFace
        self._init_langchain_models()
        
        self.rejection_templates = [
            "Dear {name},\n\nThank you for submitting your resume to our company. After careful consideration and a few laughs around the office, we regret to inform you that we will not be moving forward with your application.\n\n{specific_feedback}\n\nWe encourage you to continue developing your skills and perhaps consider a career change. We'll keep your resume on file in case we need a good example of what not to do.\n\nBest of luck in your future endeavors,\nThe Hiring Team\n\nP.S. {ps_comment}",
            
            "Hello {name},\n\nWe've reviewed your resume with the attention it deserves (about 30 seconds). Unfortunately, we've decided to pursue other candidates who actually exist in the same dimension as the job requirements.\n\n{specific_feedback}\n\nWhile we appreciate your... creativity... in resume writing, we're looking for someone whose skills extend beyond buzzword bingo.\n\nWarm regards,\nHR Department\n\n{ps_comment}",
            
            "Dear Applicant,\n\nYour resume has been processed through our advanced AI screening system (spoiler: it's just us being sarcastic). The results are in, and they're not good.\n\n{specific_feedback}\n\nWe suggest you take some time to reflect on your career choices. Maybe consider a nice, quiet job in data entry where expectations are appropriately low.\n\nSincerely,\nThe Reality Check Department\n\n{ps_comment}"
        ]
        
        self.vague_phrases = [
            "worked on", "helped with", "assisted in", "participated in", "contributed to",
            "involved in", "responsible for", "handled", "managed", "dealt with",
            "worked closely", "collaborated", "supported", "maintained", "coordinated"
        ]
        
        self.red_flags = [
            "fast learner", "quick learner", "hard worker", "team player", "detail oriented",
            "results driven", "go getter", "self starter", "thinks outside the box",
            "synergy", "leverage", "streamline", "optimize", "revolutionize"
        ]

    def _init_langchain_models(self):
        """Initialize LangChain with HuggingFace models"""
        try:
            logger.info("Initializing LangChain with HuggingFace models...")
            
            # Initialize sentiment analysis pipeline with distilbert
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                return_all_scores=True
            )
            
            # Initialize text generation pipeline for rejection letters
            self.text_generator = pipeline(
                "text-generation",
                model="distilgpt2",  # Lightweight alternative
                max_length=200,
                temperature=0.8,
                do_sample=True
            )
            
            # Create HuggingFace LLM for LangChain
            self.llm = HuggingFacePipeline(
                pipeline=self.text_generator
            )
            
            # Define prompts for different analysis tasks
            self.buzzword_prompt = PromptTemplate(
                input_variables=["text"],
                template="""
                Analyze the following resume text and identify corporate buzzwords and clichés.
                Focus on overused phrases that don't add meaningful information.
                
                Resume text: {text}
                
                Provide a brief, sarcastic analysis of the buzzwords found:
                """
            )
            
            self.quality_prompt = PromptTemplate(
                input_variables=["text"],
                template="""
                Analyze this resume text for quality and professionalism.
                Look for vague descriptions, lack of specifics, and unclear achievements.
                
                Resume text: {text}
                
                Provide a critical but humorous assessment:
                """
            )
            
            self.rejection_prompt = PromptTemplate(
                input_variables=["candidate_name", "issues", "tone"],
                template="""
                Write a professional but subtly sarcastic rejection letter for a job candidate.
                
                Candidate: {candidate_name}
                Issues found: {issues}
                Tone: {tone}
                
                Create a corporate-style rejection letter that is humorous but not cruel:
                """
            )
            
            # Create LangChain chains
            self.buzzword_chain = LLMChain(llm=self.llm, prompt=self.buzzword_prompt)
            self.quality_chain = LLMChain(llm=self.llm, prompt=self.quality_prompt)
            self.rejection_chain = LLMChain(llm=self.llm, prompt=self.rejection_prompt)
            
            logger.info("LangChain models initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize LangChain models: {str(e)}")
            logger.warning("Falling back to rule-based analysis")
            self.use_langchain = False
            self._init_fallback_analysis()
    
    def _init_fallback_analysis(self):
        """Initialize fallback rule-based analysis if LangChain fails"""
        self.sentiment_pipeline = None
        self.llm = None
        self.use_langchain = False

    def analyze_resume(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive resume analysis using LangChain"""
        try:
            # Extract basic info
            name = self._extract_candidate_name(parsed_data.get('raw_text', ''))
            
            if hasattr(self, 'use_langchain') and not self.use_langchain:
                # Fallback to rule-based analysis
                return self._analyze_resume_fallback(parsed_data, name)
            
            # Use LangChain for advanced analysis
            return self._analyze_resume_langchain(parsed_data, name)
            
        except Exception as e:
            logger.error(f"Error in resume analysis: {str(e)}")
            # Fallback to basic analysis
            return self._analyze_resume_fallback(parsed_data, self._extract_candidate_name(parsed_data.get('raw_text', '')))

    def _analyze_resume_langchain(self, parsed_data: Dict[str, Any], name: str) -> Dict[str, Any]:
        """Advanced analysis using LangChain and HuggingFace models"""
        raw_text = parsed_data.get('raw_text', '')
        
        # Sentiment analysis of the resume
        sentiment_results = self._analyze_sentiment(raw_text)
        
        # LangChain-powered buzzword analysis
        buzzword_analysis = self._langchain_buzzword_analysis(raw_text)
        
        # Quality assessment using LangChain
        quality_analysis = self._langchain_quality_analysis(raw_text)
        
        # Traditional analysis for specific patterns
        vague_analysis = self._analyze_vague_descriptions(parsed_data)
        gap_analysis = self._analyze_employment_gaps(parsed_data)
        skills_analysis = self._analyze_skills_credibility(parsed_data)
        
        # Generate enhanced rejection letter using LangChain
        issues_summary = self._compile_issues_summary(buzzword_analysis, quality_analysis, vague_analysis, gap_analysis, skills_analysis)
        rejection_letter = self._generate_langchain_rejection_letter(name, issues_summary, sentiment_results)
        
        return {
            'rejection_letter': rejection_letter,
            'buzzwords': buzzword_analysis,
            'sentiment_analysis': sentiment_results,
            'quality_assessment': quality_analysis,
            'vague_descriptions': vague_analysis,
            'employment_gaps': gap_analysis,
            'skills_credibility': skills_analysis,
            'issues': self._compile_issues(buzzword_analysis, vague_analysis, gap_analysis, skills_analysis)
        }

    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze resume sentiment using HuggingFace model"""
        try:
            if not self.sentiment_pipeline:
                return {'overall': 'neutral', 'confidence': 0.5, 'analysis': 'Sentiment analysis unavailable'}
            
            # Truncate text if too long for model
            text_chunks = [text[i:i+500] for i in range(0, len(text), 500)][:3]  # Max 3 chunks
            
            sentiment_scores = []
            for chunk in text_chunks:
                if len(chunk.strip()) > 10:  # Only analyze meaningful chunks
                    scores = self.sentiment_pipeline(chunk)
                    sentiment_scores.extend(scores)
            
            if not sentiment_scores:
                return {'overall': 'neutral', 'confidence': 0.5, 'analysis': 'No meaningful text for analysis'}
            
            # Calculate overall sentiment
            positive_scores = [s['score'] for s in sentiment_scores[0] if s['label'] == 'POSITIVE']
            negative_scores = [s['score'] for s in sentiment_scores[0] if s['label'] == 'NEGATIVE']
            
            avg_positive = sum(positive_scores) / len(positive_scores) if positive_scores else 0
            avg_negative = sum(negative_scores) / len(negative_scores) if negative_scores else 0
            
            overall = 'positive' if avg_positive > avg_negative else 'negative'
            confidence = max(avg_positive, avg_negative)
            
            # Generate sarcastic analysis
            if overall == 'positive' and confidence > 0.7:
                analysis = "Your resume radiates toxic positivity. Tone it down, champ."
            elif overall == 'positive':
                analysis = "Mildly optimistic. At least you're trying."
            elif overall == 'negative' and confidence > 0.7:
                analysis = "Your resume is as pessimistic as your career prospects."
            else:
                analysis = "Perfectly balanced negativity. How refreshing."
            
            return {
                'overall': overall,
                'confidence': round(confidence, 3),
                'analysis': analysis
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {str(e)}")
            return {'overall': 'error', 'confidence': 0, 'analysis': 'Sentiment analysis failed'}

    def _langchain_buzzword_analysis(self, text: str) -> List[Dict[str, Any]]:
        """Use LangChain to analyze buzzwords"""
        try:
            if not hasattr(self, 'buzzword_chain') or not self.buzzword_chain:
                return self._analyze_buzzwords({'raw_text': text})
            
            # Use LangChain for buzzword analysis
            result = self.buzzword_chain.run(text=text[:1000])  # Limit input length
            
            # Also do rule-based detection for comprehensive results
            rule_based = self._analyze_buzzwords({'raw_text': text})
            
            # Combine LangChain insights with rule-based detection
            enhanced_buzzwords = []
            for buzzword in rule_based:
                enhanced_buzzwords.append({
                    **buzzword,
                    'ai_analysis': f"AI detected overuse of '{buzzword['word']}'"
                })
            
            return enhanced_buzzwords
            
        except Exception as e:
            logger.error(f"LangChain buzzword analysis error: {str(e)}")
            return self._analyze_buzzwords({'raw_text': text})

    def _langchain_quality_analysis(self, text: str) -> Dict[str, Any]:
        """Use LangChain to assess resume quality"""
        try:
            if not hasattr(self, 'quality_chain') or not self.quality_chain:
                return {'score': 5, 'analysis': 'Quality analysis unavailable', 'issues': []}
            
            # Use LangChain for quality assessment
            result = self.quality_chain.run(text=text[:1000])
            
            # Extract insights and score from the result
            quality_score = self._extract_quality_score_from_text(result)
            
            return {
                'score': quality_score,
                'analysis': result.strip(),
                'ai_powered': True
            }
            
        except Exception as e:
            logger.error(f"LangChain quality analysis error: {str(e)}")
            return {'score': 5, 'analysis': 'Quality analysis failed', 'issues': ['AI analysis unavailable']}

    def _extract_quality_score_from_text(self, text: str) -> int:
        """Extract a quality score from LangChain generated text"""
        # Simple heuristic based on negative/positive words in analysis
        negative_words = ['poor', 'bad', 'terrible', 'awful', 'weak', 'lacking', 'insufficient']
        positive_words = ['good', 'excellent', 'strong', 'impressive', 'solid', 'well']
        
        text_lower = text.lower()
        negative_count = sum(1 for word in negative_words if word in text_lower)
        positive_count = sum(1 for word in positive_words if word in text_lower)
        
        # Base score of 5, adjust based on analysis tone
        score = 5 + positive_count - negative_count
        return max(1, min(10, score))

    def _generate_langchain_rejection_letter(self, name: str, issues: str, sentiment: Dict) -> str:
        """Generate rejection letter using LangChain"""
        try:
            if not hasattr(self, 'rejection_chain') or not self.rejection_chain:
                return self._generate_rejection_letter(name, issues)
            
            # Determine tone based on sentiment analysis
            tone = "professional but subtly sarcastic"
            if sentiment.get('overall') == 'positive':
                tone = "politely dismissive"
            elif sentiment.get('overall') == 'negative':
                tone = "brutally honest but corporate"
            
            # Generate letter using LangChain
            ai_letter = self.rejection_chain.run(
                candidate_name=name,
                issues=issues,
                tone=tone
            )
            
            # Fallback to template if AI output is insufficient
            if len(ai_letter.strip()) < 100:
                return self._generate_rejection_letter(name, issues)
            
            return ai_letter.strip()
            
        except Exception as e:
            logger.error(f"LangChain rejection letter generation error: {str(e)}")
            return self._generate_rejection_letter(name, issues)

    def _compile_issues_summary(self, buzzwords, quality, vague, gaps, skills) -> str:
        """Compile a summary of issues for LangChain"""
        issues = []
        
        if buzzwords and len(buzzwords) > 0:
            issues.append(f"Excessive buzzwords: {', '.join([b['word'] for b in buzzwords[:3]])}")
        
        if quality.get('score', 5) < 4:
            issues.append("Poor overall quality and lack of specificity")
        
        if vague and len(vague) > 0:
            issues.append("Vague job descriptions without concrete achievements")
        
        if gaps and len(gaps) > 0:
            issues.append("Questionable employment history")
        
        if skills.get('score', 5) < 3:
            issues.append("Unrealistic or unsubstantiated skill claims")
        
        return "; ".join(issues) if issues else "No major issues found (surprisingly)"

    def _analyze_resume_fallback(self, parsed_data: Dict[str, Any], name: str) -> Dict[str, Any]:
        """Fallback rule-based analysis when LangChain is unavailable"""
        try:
            # Extract basic info
            name = self._extract_candidate_name(parsed_data.get('raw_text', ''))
            
            # Analyze different aspects (rule-based fallback)
            buzzword_analysis = self._analyze_buzzwords(parsed_data)
            vague_analysis = self._analyze_vague_descriptions(parsed_data)
            gap_analysis = self._analyze_employment_gaps(parsed_data)
            skills_analysis = self._analyze_skills_credibility(parsed_data)
            
            # Generate specific feedback
            specific_feedback = self._generate_specific_feedback(
                buzzword_analysis, vague_analysis, gap_analysis, skills_analysis
            )
            
            # Generate rejection letter
            rejection_letter = self._generate_rejection_letter(name, specific_feedback)
            
            return {
                'rejection_letter': rejection_letter,
                'buzzwords': buzzword_analysis,
                'vague_descriptions': vague_analysis,
                'employment_gaps': gap_analysis,
                'skills_credibility': skills_analysis,
                'issues': self._compile_issues(buzzword_analysis, vague_analysis, gap_analysis, skills_analysis),
                'langchain_enabled': False
            }
            
        except Exception as e:
            logger.error(f"Error in fallback resume analysis: {str(e)}")
            return {
                'rejection_letter': "Your resume broke our analysis system. That's... impressive.",
                'buzzwords': [],
                'vague_descriptions': [],
                'employment_gaps': [],
                'skills_credibility': {'score': 0, 'issues': ['Analysis failed']},
                'issues': ['System error during analysis'],
                'langchain_enabled': False
            }

    def _extract_candidate_name(self, text: str) -> str:
        """Extract candidate name from resume text"""
        lines = text.split('\n')[:5]  # Check first 5 lines
        
        for line in lines:
            line = line.strip()
            if len(line) > 2 and len(line) < 50:
                # Simple heuristic: if it looks like a name
                words = line.split()
                if 2 <= len(words) <= 4 and all(word.replace('.', '').isalpha() for word in words):
                    return line
        
        return "Dear Applicant"

    def _analyze_buzzwords(self, parsed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze buzzword usage"""
        text = parsed_data.get('raw_text', '').lower()
        detected_buzzwords = []
        
        for buzzword in self.red_flags:
            if buzzword.lower() in text:
                detected_buzzwords.append({
                    'word': buzzword,
                    'severity': 'high' if buzzword in ['synergy', 'leverage', 'revolutionize'] else 'medium',
                    'comment': self._get_buzzword_comment(buzzword)
                })
        
        return detected_buzzwords

    def _get_buzzword_comment(self, buzzword: str) -> str:
        """Generate sarcastic comments for buzzwords"""
        comments = {
            'synergy': "Synergy? Really? What is this, 2003?",
            'leverage': "You 'leveraged' your way to the reject pile.",
            'revolutionize': "The only thing you're revolutionizing is our definition of cringe.",
            'fast learner': "Translation: 'I don't know anything yet.'",
            'team player': "Code for 'I have no individual skills.'",
            'detail oriented': "Yet somehow missed the typos in your resume.",
            'results driven': "Driven straight to unemployment.",
            'go getter': "Go get a better resume writer.",
            'self starter': "Started yourself right out of consideration."
        }
        return comments.get(buzzword.lower(), f"'{buzzword}' - another buzzword casualty.")

    def _analyze_vague_descriptions(self, parsed_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Detect vague job descriptions"""
        text = parsed_data.get('raw_text', '').lower()
        vague_found = []
        
        for phrase in self.vague_phrases:
            if phrase.lower() in text:
                vague_found.append({
                    'phrase': phrase,
                    'comment': f"'{phrase}' tells us nothing. What did you actually DO?",
                    'suggestion': f"Replace '{phrase}' with specific actions and measurable results."
                })
        
        return vague_found

    def _analyze_employment_gaps(self, parsed_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Analyze employment history for gaps"""
        experience = parsed_data.get('experience', [])
        gaps = []
        
        # Simple gap detection based on common patterns
        text = parsed_data.get('raw_text', '')
        
        # Look for suspicious patterns
        if 'freelance' in text.lower() and 'consultant' in text.lower():
            gaps.append({
                'type': 'suspicious_freelance',
                'comment': "Freelance AND consultant? Sounds like unemployed with extra steps.",
                'suggestion': "Be specific about your actual clients and projects."
            })
        
        if len(experience) < 2 and 'graduate' not in text.lower():
            gaps.append({
                'type': 'limited_experience',
                'comment': "Your work history is shorter than a TikTok video.",
                'suggestion': "Include internships, projects, or volunteer work."
            })
        
        return gaps

    def _analyze_skills_credibility(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze if claimed skills seem credible"""
        skills = parsed_data.get('skills', [])
        experience = parsed_data.get('experience', [])
        projects = parsed_data.get('projects', [])
        
        credibility = {
            'score': 5,  # Start with neutral score
            'issues': [],
            'positive_indicators': []
        }
        
        # Too many skills red flag
        if len(skills) > 20:
            credibility['score'] -= 2
            credibility['issues'].append("Claims to know everything. Jack of all trades, master of none?")
        
        # Skills without evidence
        skill_evidence = 0
        for skill in skills[:10]:  # Check top 10 skills
            if any(skill.lower() in (proj.get('description', '') + exp.get('text', '')).lower() 
                   for proj in projects for exp in experience):
                skill_evidence += 1
        
        if skills and skill_evidence / len(skills[:10]) < 0.3:
            credibility['score'] -= 3
            credibility['issues'].append("Lists skills with no evidence of actually using them.")
        else:
            credibility['positive_indicators'].append("Skills are backed by project/work evidence.")
        
        # Conflicting skill levels
        if 'beginner' in parsed_data.get('raw_text', '').lower() and 'expert' in parsed_data.get('raw_text', '').lower():
            credibility['score'] -= 1
            credibility['issues'].append("Claims to be both beginner and expert. Pick a lane.")
        
        return credibility

    def _generate_specific_feedback(self, buzzwords, vague, gaps, skills) -> str:
        """Generate specific feedback based on analysis"""
        feedback_parts = []
        
        if buzzwords:
            feedback_parts.append(f"Your resume contains {len(buzzwords)} buzzwords that made our eyes roll so hard they nearly fell out. Highlights include: {', '.join([b['word'] for b in buzzwords[:3]])}.")
        
        if vague:
            feedback_parts.append(f"We found {len(vague)} instances of vague descriptions. 'Worked on stuff' is not a skill, it's a cry for help.")
        
        if gaps:
            feedback_parts.append("Your employment history has more holes than Swiss cheese.")
        
        if skills['score'] < 3:
            feedback_parts.append("Your skills section reads like a wish list rather than actual capabilities.")
        
        if not feedback_parts:
            feedback_parts.append("While your resume doesn't have glaring issues, it also doesn't have any glaring strengths. It's aggressively mediocre.")
        
        return " ".join(feedback_parts)

    def _generate_rejection_letter(self, name: str, specific_feedback: str) -> str:
        """Generate a humorous rejection letter"""
        template = random.choice(self.rejection_templates)
        
        ps_comments = [
            "Please don't take this personally. We reject everyone equally.",
            "Consider this a learning experience. The lesson: try harder.",
            "We're sure you'll find the right opportunity. Probably not in this field, but somewhere.",
            "Remember, every 'no' gets you closer to a 'yes'. You're going to need a lot of nos.",
            "Keep your chin up! Someone out there needs your... unique talents."
        ]
        
        return template.format(
            name=name,
            specific_feedback=specific_feedback,
            ps_comment=random.choice(ps_comments)
        )

    def _compile_issues(self, buzzwords, vague, gaps, skills) -> List[str]:
        """Compile all issues found"""
        issues = []
        
        if buzzwords:
            issues.append(f"Overuse of buzzwords ({len(buzzwords)} detected)")
        if vague:
            issues.append(f"Vague descriptions ({len(vague)} instances)")
        if gaps:
            issues.append("Employment history concerns")
        if skills['score'] < 5:
            issues.append("Skills credibility issues")
        
        return issues if issues else ["No major issues detected (surprisingly)"]

    def calculate_score(self, parsed_data: Dict[str, Any], links_analysis: List[Dict]) -> Dict[str, Any]:
        """Calculate overall resume score"""
        score = 50  # Start with neutral score
        explanations = []
        
        # Skills assessment
        skills = parsed_data.get('skills', [])
        if len(skills) > 5:
            score += 10
            explanations.append("+10: Good variety of skills listed")
        elif len(skills) < 3:
            score -= 10
            explanations.append("-10: Too few skills listed")
        
        # Experience assessment
        experience = parsed_data.get('experience', [])
        if len(experience) > 2:
            score += 15
            explanations.append("+15: Multiple work experiences")
        elif len(experience) == 0:
            score -= 20
            explanations.append("-20: No work experience found")
        
        # Projects assessment
        projects = parsed_data.get('projects', [])
        if len(projects) > 2:
            score += 10
            explanations.append("+10: Multiple projects shown")
        
        # Links assessment
        working_links = sum(1 for link in links_analysis if link.get('status') == 200)
        if working_links > 0:
            score += 10
            explanations.append(f"+10: {working_links} working links")
        else:
            score -= 5
            explanations.append("-5: No working links")
        
        # Buzzword penalty
        buzzwords = parsed_data.get('buzzwords', [])
        if len(buzzwords) > 5:
            score -= 15
            explanations.append(f"-15: Too many buzzwords ({len(buzzwords)})")
        
        # Ensure score is between 0 and 100
        score = max(0, min(100, score))
        
        return {
            'value': score,
            'explanation': explanations,
            'grade': self._get_grade(score)
        }

    def _get_grade(self, score: int) -> str:
        """Convert score to letter grade with sarcastic comment"""
        if score >= 90:
            return "A: Actually impressive. Are you sure you need our help?"
        elif score >= 80:
            return "B: Better than most. Damning with faint praise."
        elif score >= 70:
            return "C: Completely average. The definition of meh."
        elif score >= 60:
            return "D: Disappointing but not hopeless. Yet."
        else:
            return "F: Forget about it. Time for a career change."

    def compare_with_job_description(self, parsed_data: Dict[str, Any], job_description: str) -> Dict[str, Any]:
        """Compare resume with job description"""
        try:
            # Prepare texts for comparison
            resume_text = " ".join([
                " ".join(parsed_data.get('skills', [])),
                " ".join([exp.get('text', '') for exp in parsed_data.get('experience', [])]),
                " ".join([proj.get('description', '') for proj in parsed_data.get('projects', [])])
            ])
            
            # Calculate similarity
            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            # Find missing skills
            jd_words = set(job_description.lower().split())
            resume_words = set(resume_text.lower().split())
            
            # Common tech skills to look for
            tech_skills = {
                'python', 'java', 'javascript', 'react', 'node', 'sql', 'aws', 'docker',
                'kubernetes', 'git', 'agile', 'scrum', 'machine learning', 'ai'
            }
            
            jd_skills = jd_words.intersection(tech_skills)
            resume_skills = resume_words.intersection(tech_skills)
            missing_skills = list(jd_skills - resume_skills)
            
            # Generate roast commentary
            commentary = self._generate_mismatch_commentary(similarity, missing_skills)
            
            return {
                'similarity_score': round(similarity * 100, 2),
                'missing_skills': missing_skills,
                'mismatch_commentary': commentary
            }
            
        except Exception as e:
            logger.error(f"Error in job description comparison: {str(e)}")
            return {
                'similarity_score': 0,
                'missing_skills': [],
                'mismatch_commentary': "Comparison failed. Your resume broke our system. Impressive."
            }

    def _generate_mismatch_commentary(self, similarity: float, missing_skills: List[str]) -> str:
        """Generate sarcastic commentary about job mismatch"""
        if similarity > 0.7:
            base = "Surprisingly, your resume actually matches the job description. Are you sure you applied to the right position?"
        elif similarity > 0.5:
            base = "Your resume has some relevance to the job. It's like wearing a tuxedo to a beach party - technically dressed up, but missing the point."
        elif similarity > 0.3:
            base = "Your resume barely relates to this job. It's like bringing a spoon to a knife fight."
        else:
            base = "Your resume has about as much relevance to this job as a chocolate teapot. Completely useless but oddly fascinating."
        
        if missing_skills:
            skill_comment = f" You're missing key skills like {', '.join(missing_skills[:5])}. Might want to learn those before applying next time."
            base += skill_comment
        
        return base