"""
Evrensel AI Fiyat/Performans Analiz Motoru (Terminal Sürümü)
Tüm e-ticaret sitelerindeki ürünleri tarar, özellikleri dinamik olarak çıkarır ve AI ile sıralar.
"""

import os
import json
import time
import requests
from bs4 import BeautifulSoup
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ──────────────────────────────────────────────
# 1. SCRAPER — Linkleri topla + metin çek
# ──────────────────────────────────────────────

def urun_linklerini_topla(ana_url: str) -> list[str]:
    print("▶ Kategori sayfası taranıyor...")
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(ana_url, headers=headers, timeout=15)
    soup = BeautifulSoup(resp.content, "lxml")

    domain = "/".join(ana_url.split("/")[:3])

    # Kara liste: Kesinlikle girmemesi gereken sayfalar
    yasakli_kelimeler = [
        "servis", "iletisim", "hakkimizda", "magaza", "kategori",
        "iptal", "iade", "alim", "surec", "kampanya"
    ]

    linkler = []
    for a in soup.find_all("a", href=True):
        href = a["href"].lower()

        if any(yasakli in href for yasakli in yasakli_kelimeler):
            continue

        # Gerçek ürün linki yakalama kuralı (.html veya bol tireli linkler)
        if href.endswith(".html") or href.count("-") >= 4:
            tam_link = href if href.startswith("http") else f"{domain}{a['href']}"
            if tam_link not in linkler and tam_link != ana_url:
                linkler.append(tam_link)

    print(f"  {len(linkler)} potansiyel ürün linki bulundu.")
    return linkler


def urun_metni_cek(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=15)
    soup = BeautifulSoup(resp.content, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.extract()
    return soup.get_text(separator=" ", strip=True)[:15000]


# ──────────────────────────────────────────────
# 2. GROQ 8B — Dinamik Yapısal Veri Çıkarımı
# ──────────────────────────────────────────────

def groq_json_cikar(ham_metin: str, dinamik_ozellikler: str) -> dict | None:
    GROQ_SISTEM = f"""
    Sen bir veri mühendisliği asistanısın. Verilen ürün metninden özellikleri çıkar, SADECE JSON döndür.
    Zorunlu Anahtarlar: urun_adi, fiyat_tl (sadece rakam).
    Ekstra Aranacak Anahtarlar: {dinamik_ozellikler}.
    Eğer metinde bilgi yoksa "Belirtilmemiş" yaz.
    """
    try:
        cevap = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": GROQ_SISTEM},
                {"role": "user", "content": f"Ürün Metni:\n{ham_metin}"}
            ],
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        return json.loads(cevap.choices[0].message.content)
    except Exception as e:
        print(f"  [!] Groq JSON çıkarım hatası: {e}")
        return None


# ──────────────────────────────────────────────
# 3. GROQ 70B — Dinamik Derin Analiz (Akıllı)
# ──────────────────────────────────────────────

def yapay_zeka_analiz(urunler: list[dict]) -> dict | None:
    urun_listesi_str = json.dumps(urunler, ensure_ascii=False, indent=2)

    sistem_talimati = """
    Sen üst düzey bir e-ticaret veri analisti ve donanım uzmanısın.
    Sana gelen ürünlerin kategorisini (Mouse, Klavye, Koltuk, Telefon vb.) anla ve onları kendi donanım standartlarına göre 0 ile 100 arasında puanla.
    Fiyat/Performans oranını (Fiyatı ucuz ama donanımı iyiyse puanı yüksek olmalı) merkeze al.
    SADECE aşağıdaki JSON şablonunda yanıt ver.
    """

    prompt = f"""
    ÜRÜNLER:
    {urun_listesi_str}

    ŞABLON:
    {{
        "skorlar": [{{"index": 0, "skor": 85, "kisa_aciklama": "Harika bir F/P ürünü."}}],
        "en_iyi_index": 0,
        "en_iyi_gerekcesi": "Neden şampiyon olduğunu açıkla."
    }}
    """

    try:
        cevap = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": sistem_talimati},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        raw_json = cevap.choices[0].message.content
        return json.loads(raw_json)
    except Exception as e:
        print(f"  [!] Groq Analiz hatası: {e}")
        return None


# ──────────────────────────────────────────────
# 4. RAPOR — Terminal'e Dinamik Çıktı
# ──────────────────────────────────────────────

def rapor_yazdir(urunler: list[dict], analiz: dict) -> None:
    print("\n" + "═" * 65)
    print("  🏆  EVRENSEL FİYAT / PERFORMANS ANALİZ RAPORU")
    print("═" * 65)

    en_iyi_idx = analiz.get("en_iyi_index", 0)
    
    # 2. KALKAN: Yapay zeka saçmalar ve null (None) dönerse sıfır kabul et
    if not isinstance(en_iyi_idx, int) or en_iyi_idx >= len(urunler):
        en_iyi_idx = 0
        
    en_iyi = urunler[en_iyi_idx]
    gerekcesi = analiz.get("en_iyi_gerekcesi", "Gerekçe belirtilmedi.")

    print(f"\n★  ŞAMPİYON : {en_iyi.get('urun_adi', 'Bilinmiyor')}")
    print(f"   Fiyat    : {en_iyi.get('fiyat_tl', en_iyi.get('fiyat', '?'))} TL")
    print(f"   Gerekçe  : {gerekcesi}\n")

    print(f"{'#':<3} {'Ürün Adı':<45} {'Fiyat':>10} {'Skor':>5}")
    print("-" * 65)

    skorlar = {s["index"]: s for s in analiz.get("skorlar", [])}
    sira = sorted(range(len(urunler)), key=lambda i: skorlar.get(i, {}).get("skor", 0), reverse=True)

    for rank, idx in enumerate(sira, 1):
        u = urunler[idx]
        skor_bilgi = skorlar.get(idx, {})
        isim = u.get("urun_adi", "?")[:43]
        fiyat = str(u.get("fiyat_tl", u.get("fiyat", "?")))
        skor = skor_bilgi.get("skor", 0)
        tac = " ★" if idx == en_iyi_idx else ""
        print(f"{rank:<3} {isim:<45} {fiyat:>10} {str(skor):>4}{tac}")

    print("═" * 65 + "\n")


# ──────────────────────────────────────────────
# 5. ANA PIPELINE (Çalıştırma Bloğu)
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Testleri buradan dinamik olarak değiştirebilirsin
    HEDEF_URL = "https://www.vatanbilgisayar.com/oyuncu-mouse/"
    ARANACAK_OZELLIKLER = "DPI, Sensör, Buton Sayısı, Ağırlık"
    MAX_URUN = 5
    
    print(f"Hedef URL: {HEDEF_URL}")
    print(f"Aranan Özellikler: {ARANACAK_OZELLIKLER}\n")

    # Adım 1: Linkleri topla
    tum_linkler = urun_linklerini_topla(HEDEF_URL)
    linkler = tum_linkler[:MAX_URUN] if MAX_URUN else tum_linkler

    # Adım 2: Her ürünü tara + Groq ile çıkar
    print(f"\n▶ {len(linkler)} ürün işlenecek...\n")
    veritabani: list[dict] = []

    for i, link in enumerate(linkler, 1):
        print(f"[{i}/{len(linkler)}] {link.split('/')[-1]}")
        ham = urun_metni_cek(link)
        veri = groq_json_cikar(ham, ARANACAK_OZELLIKLER)
        
        if veri:
            veri["kaynak_link"] = link
            veritabani.append(veri)
            print(f"  ✓ {veri.get('urun_adi','?')} — {veri.get('fiyat_tl', veri.get('fiyat','?'))} TL")
        else:
            print("  ✗ Atlandı")
        time.sleep(1.5)

    # 1. KALKAN: Boşluk kontrolü
    if not veritabani:
        print("\n[!] HATA: Ürün bulunamadı! Bot koruması olabilir veya sayfa yapısı farklı. Çıkılıyor.")
        exit(1)

    # Adım 3: Groq (Llama 70B) ile derin analiz
    print(f"\n▶ Groq Llama 70B ile {len(veritabani)} ürün analiz ediliyor...")
    analiz = yapay_zeka_analiz(veritabani)

    if not analiz:
        print("\n[!] Groq analizi başarısız, ham veriler kaydediliyor.")
        with open("urunler.json", "w", encoding="utf-8") as f:
            json.dump(veritabani, f, indent=2, ensure_ascii=False)
        exit(1)

    # Adım 4: Raporu yaz
    rapor_yazdir(veritabani, analiz)

    # Adım 5: Dosyalara kaydet
    with open("urunler_ham.json", "w", encoding="utf-8") as f:
        json.dump(veritabani, f, indent=2, ensure_ascii=False)

    with open("analiz_sonucu.json", "w", encoding="utf-8") as f:
        json.dump(analiz, f, indent=2, ensure_ascii=False)

    print("✅ urunler_ham.json ve analiz_sonucu.json dosyaları oluşturuldu.")