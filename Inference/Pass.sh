curl -i -X POST "https://example.datarobot.com/predApi/v1.0/deployments/<deploymentId>/predictions?passthroughColumns=Latitude&passthroughColumns=Longitude" \
    -H "Authorization: Bearer <API key>" \
    -H "datarobot-key: <DataRobot key>" -F \
    file=@~/.home/path/to/dataset.csv
