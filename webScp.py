import requests
from bs4 import BeautifulSoup
import csv

## function for csv
def save_to_csv(course_titles, filename='course_titles.csv'):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        csv_w = csv.writer(csvfile)
        
        
        
        for i in course_titles:
            csv_w.writerow([i])
    print(f"Data saved to {filename}.")



## function for web scrapping
def scrape_courses(base_url, max_pages=8):

    ## list to store all course titles
    all_course_titles = []  
    
    for i in range(1, max_pages + 1):
        # Construct the URL for the current page
        url = f"{base_url}{i}"
        print(f"Scraping page {i}: {url}")

        ## response
        response = requests.get(url)
        
        # check if the page exists-->  status code 200
        if response.status_code != 200:
            print(f"Page {i} does not exist or cannot be accessed.")
            break  # Stop if we hit a page that doesn’t exist
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        ## 'products__list'
        products_list = soup.find(class_='products__list')

        
        if products_list:
            # findling h3
            titles = products_list.find_all('h3')
            
            for title in titles:
                title_text = title.get_text(strip=True)
                print(f"Course Title: {title_text}")
                all_course_titles.append(title_text)  # Add title to the list
        else:
            print(f"No 'products__list' container found on page {i}.")
    
    # saving the course titles to a CSV file
    save_to_csv(all_course_titles)


## function calling
base_url = 'https://courses.analyticsvidhya.com/collections?page='
scrape_courses(base_url)
