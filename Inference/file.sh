curl -i -X POST "https://example.datarobot.com/predApi/v1.0/deployments/<deploymentId>/predictions" \
    -H "Authorization: Bearer <API key>" -F \
    file=@~/.home/path/to/dataset.csv

HTTP/1.1 200 OK
Date: Fri, 08 Feb 2019 10:00:00 GMT
Content-Type: application/json
Content-Length: 60624
Connection: keep-alive
Server: nginx/1.12.2
X-DataRobot-Execution-Time: 39
X-DataRobot-Model-Cache-Hit: true
Access-Control-Allow-Methods: OPTIONS, POST
Access-Control-Allow-Credentials: true
Access-Control-Expose-Headers: Content-Type,Content-Length,X-DataRobot-Execution-Time,X-DataRobot-Model-Cache-Hit,X-DataRobot-Model-Id,X-DataRobot-Request-Id
Access-Control-Allow-Headers: Content-Type,Authorization,datarobot-key
X-DataRobot-Request-ID: 9e61f97bf07903b8c526f4eb47830a86

{
  "data": [
    {
      "predictionValues": [
        {
          "value": 0.2570950924,
          "label": 1
        },
        {
          "value": 0.7429049076,
          "label": 0
        }
      ],
      "predictionThreshold": 0.5,
      "prediction": 0,
      "rowId": 0
    },
    {
      "predictionValues": [
        {
          "value": 0.7631880558,
          "label": 1
        },
        {
          "value": 0.2368119442,
          "label": 0
        }
      ],
      "predictionThreshold": 0.5,
      "prediction": 1,
      "rowId": 1
    }
  ]
}
