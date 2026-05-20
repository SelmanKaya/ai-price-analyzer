import os
import json
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from groq import Groq
from dotenv import load_dotenv
import streamlit as st

# Sayfa genişliği ve başlığı ayarlanıyor (Kodun en başında olmalı)
st.set_page_config(page_title="AI Fiyat/Performans Motoru", page_icon="🏆", layout="wide")

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ──────────────────────────────────────────────
# 1. VERİ KAZIMA VE AYRIŞTIRMA MANTIĞI
# ──────────────────────────────────────────────

def urun_linklerini_topla(ana_url):
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(ana_url, headers=headers, timeout=15)
    soup = BeautifulSoup(resp.content, "lxml")

    domain = "/".join(ana_url.split("/")[:3])

    yasakli_kelimeler = [
        "servis", "iletisim", "hakkimizda", "magaza", "kategori",
        "iptal", "iade", "alim", "surec", "kampanya"
    ]

    linkler = []
    for a in soup.find_all("a", href=True):
        href = a["href"].lower()

        if any(yasakli in href for yasakli in yasakli_kelimeler):
            continue

        if href.endswith(".html") or href.count("-") >= 4:
            tam_link = href if href.startswith("http") else f"{domain}{a['href']}"
            if tam_link not in linkler and tam_link != ana_url:
                linkler.append(tam_link)

    return linkler


def urun_metni_cek(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=15)
    soup = BeautifulSoup(resp.content, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.extract()
    return soup.get_text(separator=" ", strip=True)[:15000]


def groq_json_cikar(ham_metin, dinamik_ozellikler):
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
        st.warning(f"Groq JSON çıkarım hatası: {e}")
        return None


def yapay_zeka_analiz(urunler):
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
        return json.loads(cevap.choices[0].message.content)
    except Exception as e:
        st.warning(f"Groq analiz hatası: {e}")
        return None


# ──────────────────────────────────────────────
# 2. GÖRSEL ARAYÜZ (STREAMLIT UI)
# ──────────────────────────────────────────────

st.title("Evrensel AI Ürün Analiz Motoru")

hedef_url = st.text_input("Analiz Edilecek Kategori Linki:", value="https://www.vatanbilgisayar.com/oyuncu-mouse/")
aranacak_ozellikler = st.text_input("Karşılaştırılacak Özellikleri Virgülle Yazın:", value="DPI, Sensör, Buton Sayısı, Ağırlık")
max_urun = st.slider("Taranacak Maksimum Ürün Sayısı:", min_value=2, max_value=20, value=5)

if st.button("Ajanı Çalıştır ve Analiz Et 🔥"):
    if not hedef_url:
        st.warning("Lütfen geçerli bir link girin.")
    else:
        veritabani = []
        analiz = None

        with st.status("Yapay Zeka Ajanı Çalışıyor...", expanded=True) as status:

            # Adım 1: Linkleri topla
            st.write("1. Kategori sayfasındaki linkler toplanıyor...")
            tum_linkler = urun_linklerini_topla(hedef_url)
            linkler = tum_linkler[:max_urun]
            st.write(f"Bulunan Ürün Sayısı: {len(tum_linkler)}. Seçilen {len(linkler)} ürün taranıyor...")

            # Adım 2: Her ürünü tara + Groq ile çıkar
            bar = st.progress(0)
            for idx, link in enumerate(linkler, 1):
                st.write(f"Kazınıyor ({idx}/{len(linkler)}): {link.split('/')[-1]}")
                ham = urun_metni_cek(link)
                veri = groq_json_cikar(ham, aranacak_ozellikler)
                if veri:
                    veri["kaynak_link"] = link
                    veritabani.append(veri)
                bar.progress(idx / len(linkler))
                time.sleep(1.5)  # main.py ile tutarlı

            # Adım 3: Boşluk kontrolü — analizden ÖNCE yapılmalı
            if len(veritabani) == 0:
                status.update(label="Hata: Ürün Bulunamadı!", state="error", expanded=True)
                st.error(
                    "Bu site muhtemelen bot koruması (Cloudflare) kullanıyor veya ürünleri "
                    "JavaScript ile sonradan yüklüyor. Lütfen Amazon, Vatan veya N11 gibi "
                    "farklı bir site deneyin."
                )
                st.stop()

            # Adım 4: Groq (Llama 70B) ile derin analiz
            st.write(f"2. Groq Llama 70B ile {len(veritabani)} ürün analiz ediliyor...")
            analiz = yapay_zeka_analiz(veritabani)

            if analiz:
                status.update(label="Analiz Tamamlandı!", state="complete", expanded=False)
            else:
                status.update(label="Analiz Başarısız!", state="error", expanded=True)

        # ──────────────────────────────────────────────
        # 3. SONUÇLARI GÖSTER
        # ──────────────────────────────────────────────

        if analiz:
            # Şampiyon paneli
            en_iyi_idx = analiz.get("en_iyi_index", 0)

            # Yapay zeka geçersiz index dönerse sıfıra al
            if not isinstance(en_iyi_idx, int) or en_iyi_idx >= len(veritabani):
                en_iyi_idx = 0

            en_iyi_urun = veritabani[en_iyi_idx]

            st.balloons()
            st.success(f"### Fiyat/Performans Şampiyonu: {en_iyi_urun.get('urun_adi')}")
            st.info(f"**Yapay Zeka Gerekçesi:** {analiz.get('en_iyi_gerekcesi')}")

            # Skor tablosu
            st.subheader("Tüm Ürünlerin F/P Sıralaması")

            tablo_verisi = []
            skorlar_dict = {s["index"]: s for s in analiz.get("skorlar", [])}

            for idx, u in enumerate(veritabani):
                skor_bilgisi = skorlar_dict.get(idx, {})
                tablo_verisi.append({
                    "Ürün Adı": u.get("urun_adi", "?"),
                    "Fiyat (TL)": u.get("fiyat_tl", u.get("fiyat", "?")),
                    "F/P Skoru": skor_bilgisi.get("skor", 0),
                    "Kısa Değerlendirme": skor_bilgisi.get("kisa_aciklama", "-"),
                })

            tablo_verisi = sorted(tablo_verisi, key=lambda x: x["F/P Skoru"], reverse=True)
            df = pd.DataFrame(tablo_verisi)
            st.dataframe(df, width=True, hide_index=True)

        else:
            st.error("Yapay zeka analizi sırasında bir hata oluştu. Lütfen tekrar deneyin.")