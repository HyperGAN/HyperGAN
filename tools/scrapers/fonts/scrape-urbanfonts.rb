#! /usr/bin/env ruby

require 'nokogiri'
require 'open-uri'
require 'uri'

URL = 'http://www.urbanfonts.com'
def get_page(page)
  pstr = "/free-fonts_page-#{page}.htm"
  # Fetch and parse HTML document
  Nokogiri::HTML(open(URI.escape(URL+pstr)))
end
def get_font(href)
  puts "Downloading #{href}"
  begin
    content = open(URI.escape(URL+href)).read
  rescue
    return nil
  end
  puts("Content of ", content.length)
  content
end

def extract_font(font, id, name, category)
  `mkdir -p fonts/#{id}`
  open("fonts/#{id}/zip.zip", 'wb') do |file|
    file << font
  end

  puts `cd fonts/#{id} && unzip -o zip.zip`
  meta = {
    "name" => name,
    "category" => category
  }
  open("fonts/#{id}/meta.json", "w") do |json|
    json << meta
  end
end

page=1
id=0
while(true) do
  page+=1
  doc = get_page(page)
  puts "### Search for nodes by css"
  doc.css('.fontinfo').each do |info|
    category = info.css('.fontsleft a')[-1]
    name = info.css('.fontsleft a')[0]
    link = info.css('.fontsright a')[-1]
    href=link.attr('href')
    font = get_font(href)
    id+=1
    extract_font(font, id, name.content, category.content) if font
  end

end

