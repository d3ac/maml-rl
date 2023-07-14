import asyncio
import threading

def set_future_done(future, result):
    future.set_result(result) 

async def main():
    # 在主线程中创建future对象
    future = asyncio.get_event_loop().create_future() 
    
    # 启动一个线程,在线程中调用set_future_done设置future结果
    threading.Thread(target=set_future_done, args=(future, ['asd',1123,{'asd':12}])).start()

    # 等待future完成
    print(await future)

asyncio.run(main())