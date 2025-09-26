from selenium import webdriver
from selenium.webdriver.common.by import By
import requests
import io
from PIL import Image
import time
import os

def get_images_from_google(wd, delay, max_images):
    def scroll_down(wd):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(delay)

    url = "https://www.google.com/search?q=brain+tumor+ct+scan&sca_esv=2a499fe3eecb2ac0&rlz=1C1AVFC_enTR1059TR1059&udm=2&prmd=ivnbtz&source=lnt&tbs=ic:gray&uds=AMwkrPuuhztSDa__4CbZuT5KNtb1p2BsgJhRI6imEowPIpqIkVd3gTFZhU2yG79hztEzwSSHdZsTwxTLiXUpYnQbUeNaeNvdZ2xBQR0YNAwgohFs0eWL7_YRUGRxQpL-9iFPA0bKwsKRuIfcMKkDTHNNXpDHYzdfBw3WHBP4YCszP64IivxUYqGbMFMHnYRq9hYVozNQ06DnJYg-bC619EnYPJg9bBcpbqSlEP28WrAo9-npQocj-A_RceKKmVV7IeA4NQOnsALlz1SoAmDI0_z_t91z30MF6PU-UigCXp31LfTUSiehbOs&sa=X&ved=2ahUKEwjC59nO5KWFAxUWRvEDHY--A_cQpwV6BAgCEA4&biw=1536&bih=695&dpr=1.25#vhid=ZLVRcFn4sDHdDM&vssid=mosaic"
    wd.get(url)

    image_urls = set()
    while len(image_urls) < max_images:
        scroll_down(wd)

        thumbnails = wd.find_elements(By.CLASS_NAME, "H8Rx8c")

        for img in thumbnails[len(image_urls):]:
            try:
                img.click()
                time.sleep(delay)
            except:
                continue

            images = wd.find_elements(By.CLASS_NAME, "YQ4gaf")
            for image in images:
                if image.get_attribute('src') and 'http' in image.get_attribute('src'):
                    image_urls.add(image.get_attribute('src'))
                    print(f"Found {len(image_urls)}")
                    if len(image_urls) == max_images:
                        return image_urls

    return image_urls


def download_image(download_path, url, file_name):
    try:
        image_content = requests.get(url).content
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file)
        file_path = os.path.join(download_path, file_name)

        with open(file_path, "wb") as f:
            image.save(f, "JPEG")

        print(f"Success - {file_name} : {url}")
    except Exception as e:
        print(f'FAILED - {file_name}: {e}')


# Chrome WebDriver'ı başlatma:
wd = webdriver.Chrome()

# İndirilecek resim sayısı ve indirme dizini
max_images = 5
download_folder = "Tumor2"

# İndirme dizini yoksa oluşturma
if not os.path.exists(download_folder):
    os.makedirs(download_folder)

# Google'dan resim URL'lerini alma işlemi
urls = get_images_from_google(wd, 1, max_images)

# Her bir URL'den resmi indirme işlemi
for i, url in enumerate(urls):
    download_image(download_folder, url, f"{i}.jpg")

# WebDriver'ı kapatma
wd.quit()
