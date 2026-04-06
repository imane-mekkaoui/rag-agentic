from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from dotenv.ipython import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain.tools import tool
from langchain.messages import HumanMessage
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.tools import create_retriever_tool



load_dotenv(override=True)


chunks = [
"Mon nom est Mohamed Youssfi, Je suis Professeur en Informatique et Intelligence artificielle",
"Je travaille à l'ENSET Mohammedia, Université Hassan II deCasablanca",
"J'ai obtenu mon doctorat d'état en 2015, Mon doctorat de troisième cycle en 1996 et mon diplôme de professeur second cycle en 1993",
"En plus de l'informatique, je suis patiené par la musique et la culture",
"J'aime aussi écrire des récits sur ma vie et j'aime la philosophie",
"Je suis originaire de Ouarzazate, une ville au sud du Maroc",
"après les études de primaire et le collège à Ouarzazate, j'ai suivi mes études de lycée technique à Marrakech",
"Après le baccalauréat, j'ai rejoint l'ENSET Mohammedia pendant 4 années d'études pour devenir Professeur de second cycle",
"J'ai travaillé à l'ENSET depuis 1993 pour y enseigner principalement l'informatique et les sciences de l''ingénieur",
"En parallèle à mon travailler de professeur, j'ai suivi mes études supérieures à la Facultés des sciences de rabat",
"où j'ai obtenu mon DEA, Doctorat de troisième cycle puis Doctorat d'Etat dans le domaine des systèmes informatiques parallèles et distribués"
]

embedding_model = OpenAIEmbeddings()

vectorstore = Chroma.from_texts(
    texts=chunks,
    collection_name="cv_tool",
    embedding=embedding_model
    )
retriever = vectorstore.as_retriever()
retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="cv_tool",
    description="Get information about Mohamed's CV"

)


@tool
def get_employee_info(name:str):
    """
     Get information about an  given employee (name, salary, seniority)
     """
    print("get_employee_info tool invoked ")
    return{"name": name, "salary":12000, "seniority":5}

@tool
def send_email(email:str, subject:str, content:str):
    """
    Send an email  with a subject and content.
    """
    print(f"Sending email to{email},subject:{subject},content:{content}")
    return "Email sent successfully sent to {email}, subject {subject}, content {content}"

llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_agent(
    model=llm,
    tools=[get_employee_info, send_email, retriever_tool],
    system_prompt="Answer to user query using provided tools"
)

#resp = agent.invoke(input={"messages": [HumanMessage("Quel est le salaire de Yassine ?")]})
#print(resp['messages'][-1].content)

