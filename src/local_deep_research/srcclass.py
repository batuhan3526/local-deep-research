"""Domain classes for research sources, shared by harvest.py (selection prior) and corpus.py (independence counting):
primary scholarly > official docs / .gov / .edu / known-library repos > GitHub issue threads > neutral > heuristic blog
farms > listed content farms > STORM-unreliable. Stdlib only, no project imports (harvest imports corpus; both import
this). Moved out of harvest.py 2026-10-18 (v3x); harvest re-exports every name, so harvest.domain_prior etc. still work.

src_class(u) -> "unreliable" | "farm" | "": what corpus.py uses. A farm / unreliable page can still be listed as a
source of a claim, but counts as 0 independent sources: a claim supported only by farms stays Orta, never Yüksek.
"""
import re
import urllib.parse

# a URL in report / model text: balanced "(...)" stay inside (Wikipedia "War_(1683–1699)" titles), a closing ")" that
# is not part of the URL does not (live run 2026-10-20: URLs cut before their ")" went into sources lists as 404s)
URL_RE = re.compile(r"https?://(?:[^\s<>()\[\]{}\"'`|\\⟨⟩]|\([^\s<>()]{0,100}\))+")


def urls_in_text(text):
    """URLs of a text in order, trailing sentence punctuation stripped"""
    return [u.rstrip(".,;:!?") for u in URL_RE.findall(text or "")]


# STORM knowledge_storm/storm_wiki/modules/retriever.py (GENERALLY_UNRELIABLE | DEPRECATED | BLACKLISTED, from
# Wikipedia:Reliable_sources/Perennial_sources), names mapped to domains 2026-10-18. Left out on purpose: arXiv, SSRN,
# bioRxiv/medRxiv, Preprints, StackExchange/StackOverflow/MathOverflow, Wikipedia (unreliable for an encyclopedia,
# primary evidence for research like this).
UNRELIABLE = frozenset("""112.ua adfontesmedia.com alternet.org amazon.com ancestry.com answers.com antiwar.com adl.org
atlasobscura.com bild.de theblaze.com blogger.com blogspot.com broadwayworld.com californiaglobe.com thecanary.co
celebritynetworth.com cesnur.org chatgpt.com cnet.com coindesk.com consortiumnews.com counterpunch.org express.co.uk
dailykos.com dailysabah.com dailywire.com discogs.com distractify.com electronicintifada.net metal-archives.com
ethnicelebs.com facebook.com familysearch.org fandom.com thefederalist.com findagrave.com findmypast.co.uk
crwflags.com flickr.com foxnews.com gawker.com gbnews.com geni.com globalsecurity.org goodreads.com order-order.com
heatst.com history.com imdb.com indymedia.org inquisitr.com ibtimes.com ibtimes.co.uk investopedia.com
jewishvirtuallibrary.org joshuaproject.net knowyourmeme.com landtransportguru.net linkedin.com livejournal.com
marquiswhoswho.com meaww.com mediabiasfactcheck.com mrc.org medium.com metal-experience.com metro.co.uk
thenewamerican.com nypost.com ngo-monitor.org theonion.com ourcampaigns.com panampost.com patheos.com
anphoblacht.com thepostmillennial.com prnewswire.com quadrant.org.au quillette.com quora.com rawstory.com reddit.com
redstate.com rollingstone.com scribd.com simpleflying.com sixthtone.com skwawkbox.org
sourcewatch.org spirit-of-metal.com sportskeeda.com starsunfolded.com statista.com tass.com tass.ru
thetruthaboutguns.com tv.com tvtropes.org twitter.com x.com urbandictionary.com venezuelanalysis.com vgchartz.com
freebeacon.com weather2travel.com westernjournal.com wegotthiscovered.com whatculture.com ukwhoswho.com
whosampled.com wikidata.org wikileaks.org wikinews.org wordpress.com worldometers.info youtube.com zdnet.com
almayadeen.net anna-news.info baike.baidu.com cgtn.com thecradle.co crunchbase.com dailycaller.com dailymail.co.uk
dailystar.co.uk theepochtimes.com frontpagemag.com thegatewaypundit.com globaltimes.cn thegrayzone.com hispantv.com
jihadwatch.org last.fm lifesitenews.com mintpressnews.com nationalenquirer.com journal-neo.org newsbreak.com
newsblaze.com newsmax.com nndb.com occupydemocrats.com radiotelevisionmarti.com oann.com republicworld.com
royalcentral.co.uk rt.com sputniknews.com thesun.co.uk takimag.com tasnimnews.com telesurtv.net unz.com vdare.com
voltairenet.org wnd.com zerohedge.com city-data.com bestgore.com breitbart.com globalresearch.ca examiner.com
famousbirthdays.com healthline.com infowars.com lenta.ru liveleak.com lulu.com mylife.com naturalnews.com
opindia.com thepointsguy.com swarajyamag.com veteranstoday.com zoominfo.com""".split())
# Education / SEO / AI-generated content farms. The second half is from the a live run selection (2026-10-18):
# generic "X vs Y in 2026" blogs whose quotes ended up as false Yüksek corroborations and noisy contradictions.
CONTENT_FARM = frozenset("""educba.com simplilearn.com javatpoint.com tutorialspoint.com upgrad.com builtin.com
techopedia.com wallstreetmojo.com corporatefinanceinstitute.com techtarget.com spiceworks.com datacamp.com
kdnuggets.com interviewbit.com scaler.com great-learning.com mygreatlearning.com projectpro.io aiwiki.ai
toolify.ai restack.io dataconomy.com saturncloud.io analyticsinsight.net marktechpost.com
ultralytics.com/glossary
mljourney.com rottenwifi.com dailydoseofds.com breakorb.com compilenrun.com quantbrainai.net npblue.com qlumina.com
validraft.net zeuspress.org universopython.com ovrwch.com mental-momentum.ai numberanalytics.com pythondatabench.com
skills.rest lobehub.com verifiedmarketresearch.com marketresearchfuture.com eureka.patsnap.com ainvest.com
fastercapital.com englishgrammarhere.com northhavenanalytics.com referentiallabs.com quantengines.com surmount.ai
aimultiple.com
createbytes.com thelinuxcode.com codewithfimi.com articsledge.com metricgate.com pdpspectra.com opodab.com sharpskill.dev
orangeava.com arithmion.com axlfi.com readmedium.com johal.in""".split())
# ^ second batch from a live run rounds 1-2 (2026-10-19): SEO blogs that paraphrase each other, 30 / 18 / 13 /
# 9 ... Yüksek clusters supported only by them
# primary scholarly sources: preprint servers, DOI resolvers, publishers, journals, proceedings, central banks; the
# lists started from one quantitative-finance run, so major medical, life-science and humanities hosts are added too
# (a topic-agnostic tool must not rank one field's journals above another's)
PRIMARY = frozenset("""doi.org oadoi.org arxiv.org ssrn.com nber.org cepr.org repec.org bis.org imf.org oecd.org
worldbank.org ecb.europa.eu federalreserve.gov springer.com sciencedirect.com elsevier.com wiley.com tandfonline.com
jstor.org nature.com science.org pnas.org frontiersin.org ncbi.nlm.nih.gov europepmc.org plos.org acm.org ieee.org
aeaweb.org cambridge.org academic.oup.com sagepub.com emerald.com informs.org journals.uchicago.edu jmlr.org
openreview.net aclanthology.org proceedings.mlr.press mlr.press neurips.cc iclr.cc aaai.org ijcai.org thecvf.com
semanticscholar.org biorxiv.org medrxiv.org hal.science openalex.org core.ac.uk zenodo.org osf.io econstor.eu
researchsquare.com preprints.org chinaxiv.org dblp.org ci.nii.ac.jp jstage.jst.go.jp dergipark.org.tr cnki.net
cyberleninka.ru elibrary.ru scielo.org scielo.br kci.go.kr cairn.info persee.fr redalyc.org dialnet.unirioja.es
journals.plos.org link.springer.com onlinelibrary.wiley.com iopscience.iop.org papers.nips.cc thelancet.com nejm.org
bmj.com jamanetwork.com cochranelibrary.com cell.com annualreviews.org muse.jhu.edu brill.com degruyter.com
openedition.org taylorfrancis.com philpapers.org ssoar.info belleten.gov.tr islamansiklopedisi.org.tr""".split())
# v3.1: primary scholarly hosts hold many independent works (one arXiv id / DOI / article = one source), so the
# per-domain page cap does not apply to them and independence counts the work, not the host (academic_key)
_WORK_SEG = re.compile(r"(?i)\.(pdf|html?|full|abstract)$")
# official docs / standards / curated Q&A / digital archives and libraries
TRUSTED = frozenset("""mdpi.com iso.org w3.org ietf.org rfc-editor.org scikit-learn.org pytorch.org numpy.org
pandas.pydata.org scipy.org python.org statsmodels.org tensorflow.org keras.io catboost.ai xgboost.ai optuna.org
mlflow.org research.google stats.stackexchange.com quant.stackexchange.com datascience.stackexchange.com
xgboost.org lightgbm.readthedocs.io davidhbailey.com europeana.eu gallica.bnf.fr""".split())
# GitHub organisations of well-known libraries: their repos, docs and issues are primary for "how does X behave"
KNOWN_GH = frozenset("github.com/" + o for o in """microsoft dmlc catboost scikit-learn pytorch pandas-dev numpy scipy
statsmodels huggingface optuna shap mlflow google-research google-deepmind facebookresearch openai apache tensorflow
keras-team sktime unit8co nixtla awslabs amazon-science skfolio polakowo hudson-and-thames stefan-jansen
quantopian""".lower().split())
_GOV = re.compile(r"\.(gov|edu|int|mil)(\.[a-z]{2})?$|\.ac\.[a-z]{2}$|\.edu\.[a-z]{2}$|\.europa\.eu$|\.gc\.ca$")
_DOCS = re.compile(r"^(docs|doc|developer|developers)\.|\.readthedocs\.io$")
# generic AI/ML blog host names: long tokens anywhere in the registrable label, short ones only at its start / end
_FARM_LONG = ("data", "quant", "python", "coding", "code", "compile", "tech", "learn", "analytic", "insight", "brain",
              "mind", "momentum", "journey", "bench", "academy", "tutorial", "guide", "guru", "ninja", "mastery",
              "daily", "dose", "signal", "algo", "engine", "neura", "genius", "hacks", "smart")
_FARM_SHORT = ("ml", "ai", "ds", "py")
_BLOGISH = re.compile(r"^/(blogs?|articles?|insights?|posts?|guides?|tutorials?|learn|resources|academy|p)(/|$)")
FARM_OK = frozenset("deeplearning.ai fast.ai realpython.com".split())  # known-good hosts the heuristic would hit


def _in(host, doms, path=""):
    """host or one of its parent domains in a domain set; 'domain/first-path-segment' entries too"""
    parts, seg = host.split("."), (path.split("/") + ["", ""])[1].lower()
    return any(d in doms or (seg and f"{d}/{seg}" in doms) for d in (".".join(parts[i:]) for i in range(len(parts) - 1)))


def farm_like(host, path):
    """Heuristic for unlisted SEO / AI-content blogs: >= 2 generic tokens in the site name (mljourney, quantbrainai,
    dailydoseofds), or 1 plus a blog-ish path, a blog. subdomain or a .ai/.io/.dev/... TLD (universopython.com/en/blog,
    mental-momentum.ai). Personal github.io / gitlab.io pages are left alone."""
    parts = host.split(".")
    if len(parts) < 2 or host.endswith((".github.io", ".gitlab.io")) or _in(host, FARM_OK):
        return False
    lab = parts[-3] if len(parts) >= 3 and len(parts[-1]) == 2 and parts[-2] in ("co", "com", "org", "net") else parts[-2]
    lab = lab.replace("-", "")
    n = sum(t in lab for t in _FARM_LONG) + sum(lab.startswith(t) or lab.endswith(t) for t in _FARM_SHORT)
    odd = parts[-1] in ("ai", "io", "dev", "tech", "app", "xyz") or parts[0] == "blog" or bool(_BLOGISH.match(path))
    return n >= 2 or (n == 1 and odd)


def domain_prior(u):
    """Domain-class prior (added to a ~0.3-0.7 cosine): primary scholarly sources > official docs / .gov / .edu /
    known-library repos > GitHub issue threads > neutral > heuristic blog farms > listed farms > STORM-unreliable."""
    p = urllib.parse.urlsplit(u)
    host, path = p.netloc.lower().removeprefix("www."), p.path
    if _in(host, UNRELIABLE) or (host == "forbes.com" and path.startswith("/sites/")):
        return -0.10
    if _in(host, CONTENT_FARM, path):
        return -0.08
    if _in(host, PRIMARY):
        return 0.08
    if _GOV.search(host) or _DOCS.search(host) or _in(host, TRUSTED) or (host == "github.com" and _in(host, KNOWN_GH, path)):
        return 0.06
    if host == "github.com" and re.match(r"/[\w.-]+/[\w.-]+/(issues|discussions)/\d+", path):
        return 0.04
    if host != "github.com" and farm_like(host, path):
        return -0.05
    return 0.0



# Scholarly repositories: author uploads and university archives (theses, working papers, article PDFs). Preferred when
# pages are selected (harvest ACADEMIC share) but, unlike PRIMARY, not proof of peer review: they never anchor a
# Yüksek label. Live comparison 2026-10-20: a dissertation on docs.lib.purdue.edu, papers on academia.edu and
# ResearchGate were the strongest evidence a competitor found, and our selection ranked them like blogs.
REPOSITORY = frozenset("researchgate.net academia.edu scholar.archive.org philarchive.org repec.org".split())
_UNI_HOST = re.compile(r"(^|\.)(edu|ac\.[a-z]{2}|edu\.[a-z]{2})$|(^|\.)uni-[\w-]+\.[a-z]{2,}$|(^|\.)univ[\w-]*\.[a-z]{2,}$")
_REPO_PATH = re.compile(r"(?i)/(bitstreams?|handle|dspace|xmlui|jspui|eprints?|etd|theses?|dissertations?|viewcontent|"
                        r"download|repository|digital(?:archive|commons)?|server/api/core)/|\.pdf$")


def scholarly(u):
    """academic(u) or a scholarly repository (REPOSITORY hosts, or a university host with a repository / PDF path):
    the sources the page selection reserves a share for. A plain university web page (news, magazine) is not one."""
    if academic(u):
        return True
    try:
        p = urllib.parse.urlsplit(u.strip())
    except (ValueError, AttributeError):
        return False
    host = p.netloc.lower().removeprefix("www.").split(":")[0]
    if not host:
        return False
    if _in(host, REPOSITORY):
        return True
    return bool(_UNI_HOST.search(host) or host.split(".", 1)[0] in ("digitalarchive", "repository", "dspace", "eprints")) \
        and bool(_REPO_PATH.search(p.path))


# Practitioner primary sources (v3.5 yield rule 2): a tool's own repository, documentation, help centre or package page
# is the primary source for what the tool does and how it is used, as a paper is for a method. Topic-agnostic: the
# shape of the URL decides, never a topic word. Live OSINT corpus 2026-10-28: 2,900 Orta clusters backed only by such
# pages (github.com repos 2,675, help.shodan.io, support.recordedfuture.com ...) earned no verified yield at all.
CODE_HOSTS = frozenset("github.com gitlab.com codeberg.org bitbucket.org".split())
PKG_HOSTS = frozenset("pypi.org npmjs.com crates.io pkg.go.dev hub.docker.com rubygems.org packagist.org".split())
_PRAC_HOST = re.compile(r"^(help|support|kb|wiki|manual|docs?|developers?|api)\.")
_PRAC_PATH = re.compile(r"(?i)^/([a-z]{2}(-[a-z]{2})?/)?(docs?|documentation|manual|reference|api|help|support|kb|"
                        r"knowledge-?base|wiki|handbook|readme)(/|$|\.)")
_REPO_PAGE = re.compile(r"^/[\w.-]+/[\w.-]+(/(blob|tree|wiki|releases|raw)(/|$)|/?$)")  # not issues / pulls / user pages


def practitioner(u):
    """True for a practitioner primary source: a code repository page (README / blob / tree / wiki / releases on
    GitHub, GitLab, Codeberg, Bitbucket), a package registry page, or documentation / help-centre / knowledge-base
    pages (docs.*, help.*, support.* hosts; /docs, /help, /kb ... paths). Never a farm or unreliable host."""
    try:
        p = urllib.parse.urlsplit(u.strip())
        host = p.netloc.lower().removeprefix("www.").split(":")[0]
        if not host or domain_prior(u.strip()) < 0:
            return False
    except (ValueError, AttributeError):
        return False
    if host in CODE_HOSTS:
        return bool(_REPO_PAGE.match(p.path))
    return host in PKG_HOSTS or bool(_DOCS.search(host) or _PRAC_HOST.match(host) or _PRAC_PATH.match(p.path))


def src_class(u):
    """'unreliable' (STORM list, forbes.com/sites contributor posts), 'farm' (listed or heuristic SEO / AI-content
    blog) or '' -- the same thresholds as domain_prior, without its positive classes."""
    try:
        pr = domain_prior(u.strip()) if isinstance(u, str) else 0.0
    except ValueError:  # malformed URL (bad IPv6 literal...): no class, the caller decides
        pr = 0.0
    return "unreliable" if pr <= -0.10 else "farm" if pr < 0 else ""


def low_quality(u):
    return bool(src_class(u))


def academic(u):
    """True for a primary scholarly host (PRIMARY): arXiv, DOI resolvers, publishers, journals, proceedings, central
    bank / IMF / OECD working papers, regional scholarly indexes (CNKI, J-STAGE, CyberLeninka, DergiPark, SciELO)."""
    try:
        host = urllib.parse.urlsplit(u.strip()).netloc.lower().removeprefix("www.")
    except (ValueError, AttributeError):
        return False
    return bool(host) and _in(host, PRIMARY)


_ARXIV = re.compile(r"(?<![\d.])(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)")
_DOI_IN = re.compile(r"(?i)\b(10\.\d{4,9}/[^\s?#]+?)(?:\.pdf)?/?$")


def academic_key(u):
    """Independence key of a scholarly page: arXiv id / DOI when the URL names one, else host + the last meaningful
    path segment (sciencedirect .../pii/X and .../abs/pii/X are one work); None for a non-academic URL."""
    if not academic(u):
        return None
    p = urllib.parse.urlsplit(u.strip())
    host, path = p.netloc.lower().removeprefix("www."), urllib.parse.unquote(p.path)
    if "arxiv" in host:
        m = _ARXIV.search(path)
        if m:
            return "arxiv:" + m.group(1)
    m = _DOI_IN.search(path)
    if m:
        return "doi:" + m.group(1).lower()
    segs = [_WORK_SEG.sub("", x) for x in path.split("/") if x]
    segs = [x for x in segs if x and x.lower() not in ("abs", "pdf", "full", "abstract", "article", "html", "view",
                                                         "download", "epdf", "fulltext", "doi", "content")]
    ids = sorted(x for x in p.query.split("&") if re.match(r"(?i)\w*(id|doi|abstract|article|paper|doc)\w*=", x))
    return f"w:{host}/{segs[-1].lower() if segs else ''}" + ("?" + "&".join(ids).lower() if ids else "")

