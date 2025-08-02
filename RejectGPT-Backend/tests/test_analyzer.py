import unittest
from services.analyzer import ResumeAnalyzer

class TestResumeAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = ResumeAnalyzer()
    
    def test_analyze_buzzwords(self):
        parsed_data = {
            'raw_text': 'I am a passionate team player with synergy and leverage'
        }
        buzzwords = self.analyzer._analyze_buzzwords(parsed_data)
        self.assertGreater(len(buzzwords), 0)
        self.assertTrue(any(b['word'] == 'synergy' for b in buzzwords))
    
    def test_calculate_score(self):
        parsed_data = {
            'skills': ['Python', 'JavaScript', 'React'],
            'experience': [{'text': 'Software Engineer at Company'}],
            'projects': [{'description': 'Built a web app'}],
            'buzzwords': ['passionate']
        }
        links_analysis = [{'status': 200}]
        
        result = self.analyzer.calculate_score(parsed_data, links_analysis)
        self.assertIn('value', result)
        self.assertIn('explanation', result)
        self.assertIn('grade', result)
        self.assertIsInstance(result['value'], int)
        self.assertGreaterEqual(result['value'], 0)
        self.assertLessEqual(result['value'], 100)

if __name__ == '__main__':
    unittest.main()