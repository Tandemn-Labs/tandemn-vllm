import asyncio
import psutil
import iroh
from iroh.iroh_ffi import uniffi_set_event_loop

SHARED_TICKET = "docaaacay5xtknjobhxsm7jkkjp3hvag3kipi5qr42yz3bzmlhbsfumfjswah7q4osle7yh6u3rbldcz3qvm4tlbirhdlnite4sim64gwb4hespeajdnb2hi4dthixs65ltmuys2mjoojswyylzfzuxe33ifzxgk5dxn5zgwlrpamaavqca7l524ayaqj7p6sfeuqbabat675epxlqd"

async def main():
    uniffi_set_event_loop(asyncio.get_running_loop())

    options = iroh.NodeOptions()
    options.enable_docs = True
    node = await iroh.Iroh.memory_with_options(options)
    peer_id = await node.net().node_id()

    print(f"Joined as peer: {peer_id}")

    # Join the shared doc
    doc_ticket = iroh.DocTicket(SHARED_TICKET)
    doc = await node.docs().join(doc_ticket)
    print("Joined doc")

    author = await node.authors().create()
    print("Created author", author)


    key = peer_id.encode()
    print("Created key", key)
    while True:
        cpu = psutil.cpu_percent(interval=1)
        ram = psutil.virtual_memory().percent
        value = f"CPU: {cpu}%\nRAM: {ram}%".encode()
        print("Created value", value)
        try:
            await doc.set_bytes(author, key, value)
            print(f"Sent metrics as {peer_id}")
        except Exception as e:
            print(f"Failed to send metrics: {e}")

        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
