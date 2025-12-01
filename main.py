import httpx

from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIChatModel(
    'mistral-small3.2:24b-instruct-2506-q8_0',
    provider=OpenAIProvider(
        base_url='https://chat.ai.e-infra.cz/api',
        api_key="YOUR-API-KEY",
    ),
)



agent = Agent(model)

kopretina = httpx.get("https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.zahradnictvi-flos.cz%2Fadmin%2Fupload%2Fimages-cache%2F67356%2F1280.jpg%3Fv%3Da1162b92ee2366ea&f=1&nofb=1&ipt=a85da37ae3fbc0c8c90fc8d89203431fda31cdd383b7520e91ac4a24e5e7d139")
kopretina_test = httpx.get("https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.nasezahrada.com%2Fwp-content%2Fuploads%2F2023%2F06%2Fsedmikraska-uvod.jpg&f=1&nofb=1&ipt=ebb17a29585596bed73158ac0f45205510f5ddfb8155de302f31fd092d5e5626")
ruze = httpx.get("https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fcdn.myshoptet.com%2Fusr%2Fwww.kvetinybrno.cz%2Fuser%2Fshop%2Fbig%2F103-1_ruda-ruze-explorer.jpg%3F637cca27&f=1&nofb=1&ipt=eab11d40138604f92cb9463a093d43bb12bd083cd187b02314b752cbe0aed93b")

result = agent.run_sync(
    [
        'Toto je kategorie A',
        BinaryContent(data=kopretina.content, media_type='image/png'),
        "toto je kategorie B",
        BinaryContent(data=ruze.content, media_type='image/png'),
        "co je toto za kategorii?",
        BinaryContent(data=kopretina_test.content, media_type='image/png'),  
    ]
)
print(result.output)
print("##############################################")

# probably already deleted from crtlv page, u need to create yours own pictures
obrazek_1_img = httpx.get("https://ctrlv.link/shots/2025/11/01/1O2w.png")
obrazek_2_img= httpx.get("https://ctrlv.link/shots/2025/11/01/VO5M.png")

result = agent.run_sync(
    [
    "Obrázek 1 ukazuje modrý kruh a červený čtverec.",
    BinaryContent(data=obrazek_1_img.content, media_type="image/png"),
    "Obrázek 2 ukazuje, že čtverec je umístěn VLEVO a kruh je umístěn VPRAVO.",
    BinaryContent(data=obrazek_2_img.content, media_type="image/png"),
    "Na základě VLASTNOSTÍ (barev) z Obrázku 1 a POLOHY z Obrázku 2: Který z objektů je červený a je umístěn Vlevo? Odpověz POUZE názvem tvaru (Kruh nebo Čtverec)."
]
)
print(result.output)
