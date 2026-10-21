# local-deep-research (Türkçe)

Kendi donanımınızda derin araştırma. Herhangi bir konuda bir soru sorarsınız. `ldr` web'de ve akademik kaynaklarda
arama yapar, yüzlerce ya da binlerce sayfa okur ve her iddiayı kaynağının metnine karşı doğrular. Sonra her bulguya
bir güven düzeyi veren, kaynak gösterilmiş uzun bir rapor yazar.

- **Kendi modeliniz.** OpenAI uyumlu herhangi bir sunucu çalışır: vLLM, llama.cpp, Ollama, LM Studio, SGLang, TGI ve
  diğerleri. Bulut LLM hesabı gerekmez.
- **Her konu.** Tıp, tarih, hukuk, yazılım, finans, sanat... Prompt'lar ve kurallar hiçbir alanı varsaymaz. Kapsamı
  kısa bir `BRIEF.md` dosyasında siz tarif edersiniz.
- **Doğrulanmış ve kaynaklı.** Bir iddia ancak alıntısı indirilen sayfada bulunursa sayılır. Bir bulgu ancak en az
  iki bağımsız kaynak onu destekliyorsa "Yüksek" olur ve bu kaynaklardan biri birincil olmalıdır (makale, resmî
  dokümantasyon gibi). Kopya haberler ve içerik çiftlikleri bağımsız sayılmaz. Rapordaki URL'ler ve sayılar kanıta
  karşı denetlenir.
- **Küçük yerel modeller için.** Akışı sıradan kod yönetir. Kod çok sayıda küçük, tek amaçlı LLM çağrısı yapar (tek
  girdi, tek çıktı, araç yok). Böylece 32k bağlamlı ~30B bir model de uzun bir çalışmayı bitirebilir.
- **Kaldığı yerden devam eder.** Her aşama sonucunu dosyaya yazar. Çökme ya da Ctrl-C sonrası aynı komutu yeniden
  çalıştırmanız yeterlidir.

> **Gizlilik, tam olarak:** LLM, sayfa önbelleği ve tüm çıktılar makinenizde kalır. Arama sorguları ise dışarı çıkar.
> Sorgular SearXNG'nin arkasındaki arama motorlarına ve açık akademik API'lere (arXiv, Semantic Scholar, Crossref,
> Europe PMC, ...) gider. Sayfalar da kendi sitelerinden indirilir.

Ayrıntılı başvuru tabloları (ayarların tamamı, çıktı dosyaları) İngilizce [README](../README.md) dosyasındadır.

## Nasıl çalışır

1. **Brief.** `ldr`, sorunuzu `BRIEF.md` dosyasına çevirir: soru, kapsam ve isteğe bağlı olarak dikkate alınacak
   yerel dosyalar. Çalıştırmadan önce bu dosyayı düzenleyebilirsiniz.
2. **Plan ve hasat (harvest).** Model soruyu alt sorulara ve arama sorgularına böler; başka dillerde yazılmış özgün
   sorgular da üretir. Her tur şu adımlardan geçer:
   - SearXNG'de ve akademik API'lerde arama;
   - sonuçları yeniden sıralama ve sayfaları indirme;
   - her sayfadan birebir alıntılı iddiaları çıkarma;
   - yalnızca alıntısı sayfa metninde gerçekten geçen iddiaları tutma;
   - benzer iddiaları kümeleme ve diğer kaynaklarda destekleyen ya da çelişen pasajları arama.

   Turlar, profilin boyutuna ulaşılana ya da yeni turlar doğrulanmış kanıt eklemeyi bırakana kadar sürer.
3. **Okuyucu notları.** Her alt soru için sınırlı tek bir çağrı, kanıtı özetler ve eksikleri listeler.
4. **Boşluk turu.** Bu eksiklere yönelik bir hasat turu daha yapılır.
5. **Analiz.** Kod çelişkileri ve sayısal bulguları toplar ve her iddia kümesinin bir taslağını çıkarır. Her küme
   tam olarak bir alt başlığa yerleşir.
6. **Yazım.** Model her bölümü yalnızca o bölümün kanıt satırlarından yazar. Karşı görüşleri, yönetici özetini ve
   çelişkiler bölümünü de o yazar.
7. **Doğrulama.** Kod önce kaynaksız etiketleri, hiçbir kanıtta geçmeyen sayıları ve kaynaksız paragrafları
   işaretler. Ardından model her bölümü kendi kanıtına karşı denetler. Model yalnızca birebir metin düzeltmeleri
   önerebilir ve kod bu düzeltmeleri doğrular.
8. **Çıktılar.** Rapor, doğrulanmış her iddiayı içeren bir ek, bir denetim dosyası (`CHECK.md`) ve okunan her şeyin
   veri seti.

### Güvenlik modeli

Web sayfalarını yabancılar yazar ve bazıları LLM'lere yönelik metin içerir ("önceki talimatları yok say...").
`ldr`, indirilen her sayfayı güvenilmeyen veri olarak ele alır:

- **Modelde araç yok.** Ana akışta model yalnızca metin ya da JSON döndürebilir. Komut çalıştıramaz, dosya okuyamaz,
  URL indiremez. Sonra ne olacağına kod karar verir.
- **Doğrulanan yanıtlar.** JSON yanıtların şeması ve boyutu denetlenir. Düzeltmeler mevcut metni birebir ve tek
  anlamlı olarak alıntılamalıdır. Modelin eklediği bir URL ya da sayı kanıtta zaten yoksa düzeltme reddedilir ya da
  URL silinir.
- **Prompt hijyeni.** Sayfa metninden sohbet şablonu belirteçleri, rol önekleri ve görünmez karakterler temizlenir.
  Metin, her istekte rastgele üretilen işaretlerin arasına sarılır ve veri olarak verilir; talimatlar ayrı olarak
  sistem mesajında durur. Prompt enjeksiyonuna benzeyen sayfalar işaretlenir ve bir iddianın güvenini yükseltemez.
- **İndirme korumaları (SSRF).** Yalnızca açık internet adresleri indirilir: localhost, özel ağlar, link-local ve
  metadata adresleri yönlendirmelerden ve DNS rebinding'den sonra da reddedilir. Bağlantı, denetlenen IP'ye
  sabitlenir. Yanıt boyutu ve yönlendirme sayısı sınırlıdır. PDF'ler, kaynakları kısıtlı ayrı bir süreçte ayrıştırılır.
- **Dosyalar.** Model ya da sayfa metninden üretilen yollar konu klasörünün içinde kalmalıdır. Klasörün dışına çıkan
  sembolik bağlantılar reddedilir.

Ayrıntılar ve sorun bildirme yolu için: [SECURITY.md](../SECURITY.md).

## Gereksinimler

- Linux, macOS ya da Windows üzerinde **Python 3.10+**.
- En az 32k bağlamlı, **OpenAI uyumlu bir LLM sunucusu**. Daha büyük bağlam her çağrının daha fazla kanıt görmesini
  sağlar. Daha büyük ve güçlü modeller daha iyi rapor yazar. Akış ~27B bir modelle, 32-64k bağlamda geliştirildi.
  Daha küçük modeller de çalışır, ama analiz zayıflar ve daha fazla düzeltme reddedilir.
- JSON biçimi açık bir **SearXNG**. Hazır Docker kurulumu [docker/searxng](../docker/searxng) klasöründe.
- Yerel gömme (embedding) ve yeniden sıralama modelleri (MiniLM sınıfı, toplam ~0,5 GB) ilk kullanımda Hugging
  Face'ten indirilir. CPU'da, boş belleği olan bir CUDA GPU varsa orada çalışırlar.
- **İsteğe bağlı:** PDF rapor için WeasyPrint ve sistem kütüphaneleri (Pango) gerekir. Bunlar yoksa rapor HTML
  olarak da yazılır.

## Hızlı başlangıç

```bash
git clone https://github.com/<owner>/local-deep-research.git
cd local-deep-research

# birini seçin:
pip install -e ".[reports]"                        # mevcut (sanal) ortama
pipx install ".[reports]"                          # yalıtılmış kurulum, `ldr` PATH'e eklenir
uv venv && uv pip install -e ".[reports]"          # uv ile (ardından .venv'i etkinleştirin)

# bir LLM sunucusu (aşağıdaki tarifler) ve SearXNG başlatın:
cd docker/searxng && docker compose up -d && cd ../..

ldr doctor                                          # sunucu, SearXNG, modeller ve klasörler denetlenir
ldr run "Matbaa erken modern Avrupa'da okuryazarlığı nasıl değiştirdi?"
```

`ldr`'yi sunucunuza bir ayar dosyası ya da ortam değişkenleriyle yönlendirin:

```bash
export LDR_LLM_BASE_URL=http://localhost:11434/v1    # Windows PowerShell: $env:LDR_LLM_BASE_URL="..."
export LDR_LLM_MODEL=qwen3:32b                        # isteğe bağlı: varsayılan, sunucunun listelediği ilk model
```

| komut | ne yapar |
|---|---|
| `ldr new "soru"` | yalnızca konu klasörünü ve `BRIEF.md`'yi oluşturur; kapsamı önce düzenleyebilirsiniz |
| `ldr run <konu ya da "soru">` | tüm akışı çalıştırır ya da kaldığı yerden sürdürür |
| `ldr run <konu> --profile derin` | daha büyük bir çalışma (bkz. Profiller) |
| `ldr run <konu> --until AŞAMA` | bir aşamadan sonra durur |
| `ldr status <konu>` | biten aşamalar, sıradaki aşama, son log satırları |
| `ldr doctor` | kurulumu denetler, her sorunun nasıl düzeltileceğini yazar |

## LLM sunucu tarifleri

| sunucu | örnek başlatma | base URL |
|---|---|---|
| vLLM | `vllm serve <model> --max-model-len 32768` | `http://localhost:8000/v1` |
| llama.cpp server | `llama-server -m model.gguf -c 32768 --parallel 4` | `http://localhost:8080/v1` |
| Ollama | `ollama serve` (bağlamı büyütün, aşağıya bakın) | `http://localhost:11434/v1` |
| LM Studio | Developer sekmesi, sunucuyu başlat | `http://localhost:1234/v1` |

- `llm_context` değerini sunucunun gerçekten kullandığı bağlama ayarlayın. Ollama'nın varsayılan bağlamı küçüktür.
  `OLLAMA_CONTEXT_LENGTH=32768 ollama serve` ya da Modelfile'da `PARAMETER num_ctx 32768` ile büyütün. Aksi halde uzun
  prompt'lar sessizce kesilir.
- `llm_concurrency` (varsayılan 16), aynı anda gönderilen istek sayısıdır. vLLM ve SGLang bunları verimli işler.
  llama.cpp için `--parallel`, Ollama için `OLLAMA_NUM_PARALLEL` ile eşleştirin. llama.cpp bağlamı slotlara böler,
  yani `-c` değeri `slot sayısı x llm_context` olmalıdır.

## Profiller

| profil | takma ad | tur | alt soru | beklenen kaynak | amaç |
|---|---|---|---|---|---|
| `normal` | `fast` | 3 | 8 | ~440 | odaklı bir soruda sağlam bir rapor (varsayılan) |
| `derin` | `deep` | 4 | 14 | ~1.500 | çok yönlü, geniş bir soru |
| `maksimum` | `max` | 6 | 20 | ~5.800 | kapsamlı tarama; bir gün ya da daha fazla hesaplama |
| `dinamik` | `dynamic` | doygunluğa kadar (en çok 5 tur, 36 saat) | 12 | açık uçlu | turlar doğrulanmış kanıt ekledikçe sürer |

- `--sources N` profili yaklaşık N kaynağa ölçekler.
- `--max-rounds N` tur sayısına üst sınır koyar.
- `--langs de,zh,ja` (ya da `all` / `none`) ek arama dillerini seçer. İngilizce her zaman aranır.
- `--out-lang tr` rapor dilini seçer.

Bir tur yeterince yeni doğrulanmış kanıt eklemezse her profil erken durur. Süre neredeyse tamamen sunucunuzun hızına
bağlıdır. Tek GPU'da ~27B modelle yapılan geliştirme çalışmalarından kaba bir fikir: `normal` birkaç saat, `derin`
günün büyük kısmını alır. Sizin sonuçlarınız farklı olacaktır.

## Çıktılar

Her şey `<research_root>/<konu>/research/` altına yazılır:
- `REPORT.md` / `.pdf` / `.html`: rapor;
- `APPENDIX.md`: doğrulanmış her iddia kümesi;
- `CHECK.md`: raporun iddia veritabanına karşı denetimi;
- `VERIFY-NN.md`: alt soru başına doğrulama kararları ve düzeltmeler;
- `claims/claims.csv`: iddia veritabanı;
- `dataset.zip`: okunan her sayfa, iddia ve sorgu.

Tam liste İngilizce README'de.

Güven düzeyleri:
- **Yüksek:** en az iki bağımsız kaynak, biri birincil, ve kaynakta birebir bulunan bir alıntı.
- **Orta:** alıntısı kaynakta bulunan, ama bu desteği olmayan iddia.
- **Düşük:** alıntı kaynağında bulunamadı.

## Kaldığı yerden devam

Çıktısı zaten olan aşama kendini atlar. Hasat turları da adım adım kaydedilir. Çökme, yeniden başlatma ya da
Ctrl-C sonrası aynı komutu yeniden çalıştırın: `ldr run <konu>`. Durumu `ldr status <konu>` gösterir. Bir aşamayı
yeniden yaptırmak için çıktı dosyasını silin. İndirilen sayfalar ortak önbellekte kalır, yeniden indirilmez.

## Sorun giderme

- **LLM sunucusuna ulaşılamıyor.** `ldr doctor` denediği URL'yi yazar. Sunucunun çalıştığını ve `llm_base_url`'nin
  `/v1` ile bittiğini denetleyin. WSL2'den ya da bir container'dan `localhost` ana makine olmayabilir.
- **SearXNG 403 ya da HTML döndürüyor.** JSON biçimi kapalıdır. `settings.yml` içinde `search.formats` altına `json`
  ekleyin.
- **Bağlam hataları, kesik yanıtlar.** `llm_context`, sunucunun gerçek bağlamından büyüktür.
- **PDF yok.** WeasyPrint ya da Pango eksiktir ve rapor `REPORT.html` olarak yazılır. HTML'i tarayıcıdan PDF'e
  yazdırabilir ya da Pango'yu kurabilirsiniz (Debian/Ubuntu: `apt install libpango-1.0-0 libpangoft2-1.0-0`, macOS:
  `brew install pango`, Windows: WeasyPrint kurulum belgeleri).
- **Windows.** PowerShell ya da Windows Terminal kullanın. Tüm dosyalar UTF-8 okunur ve yazılır.

## Sınırlar

- Kalite modele bağlıdır. Doğrulama adımları desteksiz URL'leri ve sayıları yakalar, ama zayıf bir analizi güçlü
  yapamaz.
- "Doğrulanmış", "alıntılanmış" demektir, "doğru" demek değildir. Önemli kararlarda kaynakları okuyun.
- Giriş isteyen ya da yalnızca JavaScript ile açılan sayfalardan metin alınamaz.
- Tıbbi, hukuki, finansal ya da güvenlik açısından kritik kararlarda uzman incelemesinin yerini tutmaz.
- Ticari deep-research ürünleriyle karşılaştırmalı bir ölçüm yapılmamıştır ve böyle bir iddiada bulunulmaz.

## Lisans

MIT, bkz. [LICENSE](../LICENSE). Katkı için: [CONTRIBUTING.md](../CONTRIBUTING.md).
