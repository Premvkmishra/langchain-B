import asyncio
import aiohttp
import logging
from typing import List, Dict, Any
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LinkChecker:
    def __init__(self, timeout: int = 10, max_concurrent: int = 5):
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self.witty_comments = {
            200: [
                "Link works! Surprisingly functional, unlike your career.",
                "This link is alive and well. Unlike your hopes and dreams.",
                "URL is accessible. At least something in your resume works.",
                "Link is valid. One point for you, I guess."
            ],
            404: [
                "Your link is lost in the digital void, just like your potential.",
                "404 - Not Found. Much like your job prospects.",
                "This link is as missing as your actual work experience.",
                "URL not found. Did you make this up like your skills?",
                "Dead link detected. It died faster than your last relationship."
            ],
            403: [
                "Access forbidden. Even your own portfolio doesn't want you.",
                "Permission denied. Your own website is rejecting you.",
                "Forbidden access. Your portfolio has standards, apparently.",
                "403 - The internet said 'no' to your link."
            ],
            'timeout': [
                "Your link timed out. Even the internet gave up on you.",
                "Timeout error. Your portfolio loads slower than your career progress.",
                "Request timeout. Your website is as responsive as you are to feedback.",
                "Connection timeout. Even servers don't want to talk to you."
            ],
            'error': [
                "Link error. Your URL is broken, just like your dreams.",
                "Connection failed. Your link has commitment issues.",
                "Network error. Your website is having an existential crisis.",
                "Link is unreachable. Much like your career goals.",
                "Something went wrong. Story of your life, right?"
            ]
        }

    def validate_links(self, links: List[str]) -> List[Dict[str, Any]]:
        """Validate all links and return status with witty comments"""
        if not links:
            return []
        
        try:
            # Run async link checking
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(self._check_links_async(links))
            loop.close()
            return results
        except Exception as e:
            logger.error(f"Error in link validation: {str(e)}")
            return [{'url': link, 'status': 'error', 'comment': 'Failed to check link'} for link in links]

    async def _check_links_async(self, links: List[str]) -> List[Dict[str, Any]]:
        """Asynchronously check multiple links"""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            connector=aiohttp.TCPConnector(limit=self.max_concurrent)
        ) as session:
            tasks = [self._check_single_link(session, semaphore, link) for link in links]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append({
                        'url': links[i],
                        'status': 'error',
                        'comment': self._get_random_comment('error'),
                        'error': str(result)
                    })
                else:
                    processed_results.append(result)
            
            return processed_results

    async def _check_single_link(self, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore, url: str) -> Dict[str, Any]:
        """Check a single link status"""
        async with semaphore:
            try:
                logger.info(f"Checking link: {url}")
                
                # Add some headers to avoid being blocked
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                start_time = time.time()
                
                async with session.get(url, headers=headers, allow_redirects=True) as response:
                    response_time = round((time.time() - start_time) * 1000, 2)
                    
                    result = {
                        'url': url,
                        'status': response.status,
                        'response_time': response_time,
                        'comment': self._get_random_comment(response.status)
                    }
                    
                    # Add additional info for successful requests
                    if response.status == 200:
                        result['title'] = await self._extract_title(response)
                    
                    return result
                    
            except asyncio.TimeoutError:
                return {
                    'url': url,
                    'status': 'timeout',
                    'comment': self._get_random_comment('timeout'),
                    'error': 'Request timeout'
                }
            except aiohttp.ClientError as e:
                return {
                    'url': url,
                    'status': 'error',
                    'comment': self._get_random_comment('error'),
                    'error': str(e)
                }
            except Exception as e:
                return {
                    'url': url,
                    'status': 'error',
                    'comment': self._get_random_comment('error'),
                    'error': f"Unexpected error: {str(e)}"
                }

    async def _extract_title(self, response: aiohttp.ClientResponse) -> str:
        """Extract page title from response"""
        try:
            if 'text/html' in response.headers.get('content-type', ''):
                content = await response.text()
                import re
                title_match = re.search(r'<title[^>]*>([^<]+)</title>', content, re.IGNORECASE)
                if title_match:
                    return title_match.group(1).strip()
        except:
            pass
        return "No title found"

    def _get_random_comment(self, status) -> str:
        """Get a random witty comment based on status"""
        import random
        
        if status in self.witty_comments:
            return random.choice(self.witty_comments[status])
        elif 400 <= status < 500:
            return random.choice([
                f"HTTP {status} - Client error. Fitting, considering the source.",
                f"Error {status} - Your link has issues, just like you.",
                f"Status {status} - Something's wrong, and it's probably you."
            ])
        elif 500 <= status < 600:
            return random.choice([
                f"HTTP {status} - Server error. Even servers can't handle your portfolio.",
                f"Error {status} - The server crashed looking at your work.",
                f"Status {status} - You broke the internet. Congratulations."
            ])
        else:
            return f"HTTP {status} - Unexpected status. Like your career path."

    def check_link_quality(self, url: str, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze link quality and provide feedback"""
        quality_analysis = {
            'score': 0,
            'issues': [],
            'suggestions': []
        }
        
        # Check response time
        response_time = response_data.get('response_time', 0)
        if response_time > 3000:  # 3 seconds
            quality_analysis['issues'].append("Site loads slower than your career progress")
            quality_analysis['suggestions'].append("Consider hosting somewhere faster")
        elif response_time < 1000:
            quality_analysis['score'] += 2
        
        # Check if it's a common platform
        if 'github.com' in url.lower():
            quality_analysis['score'] += 3
            if '/blob/' in url or '/tree/' in url:
                quality_analysis['suggestions'].append("Link to your profile, not a specific file")
        elif 'linkedin.com' in url.lower():
            quality_analysis['score'] += 2
        elif 'portfolio' in url.lower() or 'website' in url.lower():
            quality_analysis['score'] += 3
        
        # Check title quality
        title = response_data.get('title', '')
        if title and title.lower() not in ['untitled', 'new document', 'index']:
            quality_analysis['score'] += 1
        else:
            quality_analysis['issues'].append("Generic or missing page title")
        
        # Final score calculation
        if response_data.get('status') == 200:
            quality_analysis['score'] += 5
        elif response_data.get('status') in [403, 404]:
            quality_analysis['score'] = 0
            quality_analysis['issues'].append("Link is broken or inaccessible")
        
        # Cap score at 10
        quality_analysis['score'] = min(quality_analysis['score'], 10)
        
        return quality_analysis