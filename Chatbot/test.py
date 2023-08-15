import inspect

import WebCrawlSpotify
from songs import MockSongs

def assert_equals(expected, actual):
    if expected == actual:
        pass_test()
    else:
        fail_test()

def assert_not_equals(expected, actual):
    if not expected == actual:
        pass_test()
    else:
        fail_test()

def pass_test():
    print(inspect.stack()[2][3] + ' - pass')

def fail_test():
    print(inspect.stack()[2][3] + ' - fail')

def test_assert_equals():
    assert_equals(1,1)



def test_empty_url_returns_negative_one():
    expected = '-1'
    actual = WebCrawlSpotify.getSoupFromWebsite(None, 'n/a', 'n/a', None)
    assert_equals(expected, actual)
 




def test_handle_not_found_cannot_find():
    url = 'https://genius.com/NOTREAdsaAdwL'
    title = 'Go to Hell'
    artist = 'Letdown.'
    
    expected = '-1'
    songDb = MockSongs()
    actual = WebCrawlSpotify.handle_page_not_found(url, title, artist, songDb)

    assert_equals(expected, actual)


def test_handle_not_found_can_find():
    url = 'https://genius.com/NOTREAdsaAdwL'
    title = 'sun and moon'
    artist = 'anees'
    
    actual = WebCrawlSpotify.handle_page_not_found(url, title, artist, None)

    assert_not_equals('-1', actual)



#What do you want to test?
#What do you want to have happen?
if __name__ == '__main__':
    test_assert_equals()
    test_empty_url_returns_negative_one()
    test_handle_not_found_cannot_find()
    test_handle_not_found_can_find()