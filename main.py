import httpx
from pathlib import Path

from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models.openai import OpenAIResponsesModelSettings, OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIChatModel(
    'mistral-small3.2:24b-instruct-2506-q8_0',
    provider=OpenAIProvider(
        base_url='https://chat.ai.e-infra.cz/api',
        api_key=Path('api_key').read_text().strip(),
    ),
)



agent = Agent(model)


kopretina = httpx.get("https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.zahradnictvi-flos.cz%2Fadmin%2Fupload%2Fimages-cache%2F67356%2F1280.jpg%3Fv%3Da1162b92ee2366ea&f=1&nofb=1&ipt=a85da37ae3fbc0c8c90fc8d89203431fda31cdd383b7520e91ac4a24e5e7d139")
kopretina_test = httpx.get("https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.nasezahrada.com%2Fwp-content%2Fuploads%2F2023%2F06%2Fsedmikraska-uvod.jpg&f=1&nofb=1&ipt=ebb17a29585596bed73158ac0f45205510f5ddfb8155de302f31fd092d5e5626")
ruze = httpx.get("https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fcdn.myshoptet.com%2Fusr%2Fwww.kvetinybrno.cz%2Fuser%2Fshop%2Fbig%2F103-1_ruda-ruze-explorer.jpg%3F637cca27&f=1&nofb=1&ipt=eab11d40138604f92cb9463a093d43bb12bd083cd187b02314b752cbe0aed93b")

result = agent.run_sync(
    [
        'This is category A',
        BinaryContent(data=kopretina.content, media_type='image/png'),
        "This is category B",
        BinaryContent(data=ruze.content, media_type='image/png'),
        "What category is this?",
        BinaryContent(data=kopretina_test.content, media_type='image/png'),  
    ]
)

print(result.output)
print("##############################################")
model_settings = OpenAIResponsesModelSettings(openai_previous_response_id='auto')
result = agent.run_sync(
    user_prompt=[
        "What task did you just solve and how did you solve it?.",
    ],
    message_history=result.new_messages(),
    model_settings=model_settings,

)
print(result.output)

result = agent.run_sync(
    user_prompt=[
        "Provide me with a sequence of random floating point numbers between 0 and 1, each enclosed in curly braces. For example: {0.1234}, {0.5678}, {0.91011}. Generate 5 such numbers.",
    ],
    message_history=result.new_messages(),
    model_settings=model_settings,

)
print(result.output)
