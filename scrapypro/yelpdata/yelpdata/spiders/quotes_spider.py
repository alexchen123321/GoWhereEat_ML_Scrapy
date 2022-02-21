import scrapy


class AuthorSpider(scrapy.Spider):
    name = 'chop'

    start_urls = [
        # 'http://quotes.toscrape.com/'
        # 'https://www.chope.co/singapore-restaurants/restaurant/eight-korean-bbq?source=chope.com.sg'
        'https://www.chope.co/singapore-restaurants/list_of_restaurants?source=chope.com.sg&lang=en_US'
                  ]

    def parse(self, response):
        author_page_links = response.css('li.cf a::attr(href)')
        yield from response.follow_all(author_page_links, self.parse_author)

        # pagination_links = response.css('li.next a')
        # yield from response.follow_all(pagination_links, self.parse)

    def parse_author(self, response):
        def extract_with_css(query):
            return response.css(query).get(default='').strip()
        def extract_with_xpath(query):
            return response.xpath(query).get(default='').strip()
        yield {
            'name': extract_with_xpath('//h1/span/text()'),
            'category': extract_with_xpath('//*[@id="rstr_info"]/ul[1]/li[1]/p/text()'),
            'address': extract_with_xpath('//*[@id="rstr_info"]/ul[2]/li[2]/p/text()'),
            'opentime': extract_with_xpath('//*[@id="rstr_info"]/ul[1]/li[3]/p/text()'),
            'price': extract_with_xpath('//*[@id="rstr_info"]/ul[1]/li[4]/p/text()'),
            'goodfor': extract_with_xpath('//*[@id="rstr_info"]/ul[1]/li[5]/p/text()'),
            'map': extract_with_css('p.mapbox a::attr(href)'),
            'website':  response.request.url
        }


#
# scrapy shell 'https://www.chope.co/singapore-restaurants/restaurant/eight-korean-bbq?source=chope.com.sg'
#
# >>> response1 = response.replace(body=response.body.replace(b'<br>', b' '))
#
#
# category
# >>> response1.xpath('//*[@id="rstr_info"]/ul[1]/li[1]/p/text()').get().strip()
# 'Korean, BBQ'
#
# address
# >>> response1.xpath('//*[@id="rstr_info"]/ul[2]/li[2]/p/text()').get().strip()
# '1 Scotts Road #04-20/21 Shaw Centre Singapore (228208)'
#
# name
# >>> response1.xpath('//h1/span/text()').get()
# '8 Korean BBQ (Shaw Centre)'
#
# opentime
# >>> response1.xpath('//*[@id="rstr_info"]/ul[1]/li[3]/p/text()').get().strip()
# 'Mon-Fri: 11:30am-2:30pm, 5:30-10:30pm'
#
# price
# >>> response1.xpath('//*[@id="rstr_info"]/ul[1]/li[4]/p/text()').get()
# '$$'
#
# https://www.chope.co/singapore-restaurants/restaurant/eight-korean-bbq?source=chope.com.sg
#
# map
# >>> response1.css('p.mapbox a::attr(href)').get()
# 'https://www.google.com/maps/search/?api=1&query=1.3062411,103.8318312&query_place_id=ChIJpeavFI0Z2jEReoYE7RTQ8m0'
#
# good for
# >>> response1.xpath('//*[@id="rstr_info"]/ul[1]/li[5]/p/text()').get()
# 'Casual Dining, Kid Friendly, Drinks, Large Parties (16+)'
#
#
#
# chope.co/singapore-restaurants/restaurant/yassin-kampung-marsiling', 'https://www.chope.co/singapore-restaurants/restaurant/yechun-xiao-jiang-nan', 'https://www.chope.co/singapore-restaurants/restaurant/yellow-cab-pizza-co', 'https://www.chope.co/singapore-restaurants/restaurant/yellow-pot-little-india', 'https://shop.chope.co/products/yellow-pot-little-india-new', 'https://www.chope.co/singapore-restaurants/restaurant/yellow-pot-tanjong-pagar', 'https://shop.chope.co/products/yellow-pot-tanjong-pagar', 'https://www.chope.co/singapore-restaurants/restaurant/yi-jia-south-village-seafood-restaurant', 'https://www.chope.co/singapore-restaurants/restaurant/yin-at-the-riverhouse', 'https://www.chope.co/singapore-restaurants/restaurant/yoshi-restaurant', 'https://www.chope.co/singapore-restaurants/restaurant/you-are-my-sunshine', 'https://www.chope.co/singapore-restaurants/restaurant/youngs-bar-restaurant', 'https://www.chope.co/singapore-restaurants/restaurant/yujin-izakaya', 'https://www.chope.co/singapore-restaurants/restaurant/yum-cha-changi', 'https://www.chope.co/singapore-restaurants/restaurant/yum-cha-chinatown', 'https://www.chope.co/singapore-restaurants/restaurant/yummo-chow', 'http://shop.chope.co/products/yummo-chow?aff=25&utm_source=ChopeSiteListing&utm_medium=promo_notes&utm_campaign=ChopeVouchers', 'https://www.chope.co/singapore-restaurants/restaurant/yuugo', 'https://www.chope.co/singapore-restaurants/restaurant/zafferano', 'https://shop.chope.co/products/zafferano', 'https://www.chope.co/singapore-restaurants/restaurant/zaffron-kitchen-east-coast', 'https://shop.chope.co/products/zaffron-kitchen-east-coast', 'https://www.chope.co/singapore-restaurants/restaurant/zaffron-kitchen-great-world', 'https://www.chope.co/singapore-restaurants/restaurant/zaffron-kitchen-the-star-vista', 'https://www.chope.co/singapore-restaurants/restaurant/ziggys-lounge-rooms-cocktail-bar', 'https://www.chope.co/singapore-restaurants/restaurant/zorba-the-greek-taverna-cafe-and-restaurant', 'https://shop.chope.co/products/zorba-the-greek-taverna', 'https://www.chope.co/singapore-restaurants/restaurant/zui-yu-xuan-teochew-cuisine']
# >>> response.css('li.cf a::attr(href)').getall()
#
#
#
# scrapy shell 'https://www.chope.co/singapore-restaurants/list_of_restaurants?source=chope.com.sg&lang=en_US'
