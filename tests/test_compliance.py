from idea_miner.core.compliance import author_hash, html_to_text, scrub


def test_scrub_removes_personal_data():
    text = ("Mail me at jane.doe@example.com or call +91 98765 43210, my PAN is ABCDE1234F. "
            "Ping @jane_d or see https://twitter.com/jane_d/status/1")
    out = scrub(text)
    assert "jane.doe@example.com" not in out and "[email]" in out
    assert "98765" not in out and "[number]" in out
    assert "ABCDE1234F" not in out
    assert "@jane_d" not in out
    assert "[link:twitter.com]" in out


def test_scrub_keeps_ordinary_numbers():
    out = scrub("I spend 3 hours on 250 invoices in 2026, costs Rs 1,500")
    assert out == "I spend 3 hours on 250 invoices in 2026, costs Rs 1,500"


def test_author_hash_is_stable_and_anonymous():
    a = author_hash("hackernews", "SomeUser")
    assert a == author_hash("hackernews", "someuser")
    assert "someuser" not in a and len(a) == 16
    assert author_hash("hackernews", "") == ""


def test_html_to_text():
    assert html_to_text("<p>Hello &amp; <a href='x'>world</a></p><p>Next</p>") == "Hello & world\n\nNext"
