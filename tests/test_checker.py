import unittest
from unittest.mock import Mock, patch
from services.link_checker import LinkChecker

class TestLinkChecker(unittest.TestCase):
    def setUp(self):
        self.checker = LinkChecker()
    
    def test_get_random_comment(self):
        comment_200 = self.checker._get_random_comment(200)
        self.assertIsInstance(comment_200, str)
        self.assertGreater(len(comment_200), 0)
        
        comment_404 = self.checker._get_random_comment(404)
        self.assertIsInstance(comment_404, str)
        self.assertIn('404', comment_404)
    
    def test_validate_links_empty(self):
        result = self.checker.validate_links([])
        self.assertEqual(result, [])
