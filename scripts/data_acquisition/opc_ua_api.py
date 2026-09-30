import asyncio
from asyncua import Server, Client, ua

async def main():  # simulate PLC/SCADA endpoint
    endpoint = "opc.tcp://127.0.0.1:4840/toy/"

    server = Server()
    await server.init()
    server.set_endpoint(endpoint)

    namespace = await server.register_namespace("urn:toy:plant")
    machine = await server.nodes.objects.add_object(namespace, "Machine")

    await machine.add_variable(
        ua.NodeId("Temperature", namespace),
        "Temperature",
        72.5,
    )

    async with server:
        # This block is the basic real-world client usage.
        async with Client(endpoint) as client:
            ns = await client.get_namespace_index("urn:toy:plant")
            node = client.get_node(ua.NodeId("Temperature", ns))
            value = await node.read_value()
            print(value)  # 72.5

asyncio.run(main())