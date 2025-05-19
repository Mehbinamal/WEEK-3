from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import google.generativeai as genai
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class WebBrowserTool:
    def __init__(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        
        # Add additional preferences
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        self.driver = webdriver.Chrome(options=chrome_options)
        # Set window size
        self.driver.set_window_size(1920, 1080)

    async def fetch(self, query):
        try:
            # First try to get content from the URL if it's a direct URL
            if query.startswith(('http://', 'https://')):
                url = query
            else:
                # If it's a search query, use Google search
                url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            
            self.driver.get(url)
            
            # Wait for content to load
            wait = WebDriverWait(self.driver, 10)
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            
            # Add a small delay to ensure content is fully loaded
            await asyncio.sleep(3)
            
            # Try to get the main content
            content = ""
            
            # If it's a Google search result
            if "google.com/search" in url:
                # Try to get search results
                try:
                    search_results = self.driver.find_elements(By.CSS_SELECTOR, "div.g")
                    for result in search_results[:5]:  # Get first 5 results
                        try:
                            title = result.find_element(By.CSS_SELECTOR, "h3").text
                            snippet = result.find_element(By.CSS_SELECTOR, "div.VwiC3b").text
                            content += f"Title: {title}\nContent: {snippet}\n\n"
                        except:
                            continue
                except:
                    pass
            
            # If we don't have content from search results or it's a direct URL
            if not content:
                # Try to get the main content of the page
                try:
                    # Try common content containers
                    selectors = [
                        "article", "main", "div.content", "div.main-content",
                        "div.post-content", "div.article-content", "div.entry-content"
                    ]
                    
                    for selector in selectors:
                        try:
                            elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                            if elements:
                                content = elements[0].text
                                break
                        except:
                            continue
                    
                    # If no specific content found, get body text
                    if not content:
                        content = self.driver.find_element(By.TAG_NAME, "body").text
                except:
                    content = self.driver.find_element(By.TAG_NAME, "body").text
            
            if not content:
                return "Could not extract meaningful content from the page."
                
            return content[:5000]  # Limit content length
            
        except Exception as e:
            return f"Error fetching content: {str(e)}"
        
    def __del__(self):
        try:
            self.driver.quit()
        except:
            pass


class TextSummarizerTool:
    def __init__(self):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        self.max_retries = 3

    async def summarize(self, text):
        if not text or len(text.strip()) == 0:
            return "Error: No content provided for summarization"
            
        # Clean and prepare the text
        text = text.strip()
        if len(text) > 5000:
            text = text[:5000] + "..."
            
        prompt = f"""Please provide a concise summary of the following text. 
        Focus on the main points and key information:
        
        {text}
        
        Summary:"""
        
        for attempt in range(self.max_retries):
            try:
                response = await self.model.generate_content_async(prompt)
                if response and response.text:
                    return response.text.strip()
                else:
                    print(f"Attempt {attempt + 1}: Empty response from Gemini")
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    return f"Error: Failed to generate summary after {self.max_retries} attempts"
                await asyncio.sleep(1)  # Wait before retrying
                
        return "Error: Failed to generate summary"
