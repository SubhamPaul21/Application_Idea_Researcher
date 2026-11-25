from idea_miner.pipeline.prefilter import Prefilter


def test_matches_wish_phrases():
    pf = Prefilter()
    ok, hits = pf.evaluate("Is there an app that reminds tenants about rent automatically?")
    assert ok and "is_there_a_tool" in hits
    ok, hits = pf.evaluate("I’d happily pay for something that syncs my invoices with Tally")
    assert ok and "would_pay" in hits
    ok, hits = pf.evaluate("I'm so tired of manually copying orders from WhatsApp into a sheet")
    assert ok and "tired_of_manual" in hits


def test_rejects_noise_and_negatives():
    pf = Prefilter()
    assert pf.evaluate("Nice sunset today, sharing a picture from the beach")[0] is False
    assert pf.evaluate("We're hiring! Is there an app developer who wants to join us?")[0] is False
    assert pf.evaluate("short")[0] is False


def test_bypass_for_problem_sources():
    pf = Prefilter()
    ok, hits = pf.evaluate("The app crashes whenever I export my monthly report to PDF.", bypass=True)
    assert ok and hits == []
