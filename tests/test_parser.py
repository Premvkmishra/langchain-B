import unittest
import os
import tempfile
from services.parser import ResumeParser

class TestResumeParser(unittest.TestCase):
    def setUp(self):
        self.parser = ResumeParser()
    
    def test_extract_skills(self):
        text = "I have experience with Python, JavaScript, and React. I also know SQL and Docker."
        skills = self.parser._extract_skills(text)
        self.assertIn('python', [s.lower() for s in skills])
        self.assertIn('javascript', [s.lower() for s in skills])
        self.assertIn('react', [s.lower() for s in skills])
    
    def test_extract_links(self):
        text = "Check out my portfolio at https://example.com and my GitHub at github.com/user"
        links = self.parser._extract_links(text)
        self.assertIn('https://example.com', links)
        self.assertIn('https://github.com/user', links)
    
    def test_detect_buzzwords(self):
        text = "I'm a passionate, results-driven team player with synergy"
        buzzwords = self.parser._detect_buzzwords(text)
        self.assertIn('passionate', buzzwords)
        self.assertIn('synergy', buzzwords)