import pika
from pyduyp.config.conf import get_mq_args

args = get_mq_args()


# 发送到mq
def send(queue, body):
    credentials = pika.PlainCredentials(args.get('user'), args.get('pass'))
    connection = pika.BlockingConnection(pika.ConnectionParameters(
        args.get('host'), args.get('port'), '/', credentials))
    channel = connection.channel()

    # 声明queue
    channel.queue_declare(queue=queue)

    # n RabbitMQ a message can never be sent directly to the queue, it always needs to go through an exchange.
    ret = channel.basic_publish(exchange='',
                          routing_key=queue,
                          body=body)
    connection.close()
    return ret
