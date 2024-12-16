import csv


class HousingPrices:
    def __init__(self):
        self.data = {}  # Initialize data as an instance variable

    def saveData(self, HouseID, Price, Type, squareMeters, Bedrooms, Bathrooms, BuildYear, useableArea, SchoolDistance,
                 Condition):
        self.data[HouseID] = {
            'Price': Price,
            'Type': Type,
            'squareMeters': squareMeters,
            'Bedrooms': Bedrooms,
            'Bathrooms': Bathrooms,
            'BuildYear': BuildYear,
            'useableArea': useableArea,
            'SchoolDistance': SchoolDistance,
            'Condition': Condition  # Corrected spelling
        }

    def storeData(self):
        fieldNames = ['HouseID', 'Price', 'Type', 'squareMeters', 'Bedrooms', 'Bathrooms', 'BuildYear', 'useableArea',
                      'SchoolDistance', 'Condition']

        with open('ApartmentData.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldNames)
            writer.writeheader()

            # Write each house's data as a row
            for HouseID, house_data in self.data.items():
                # Add HouseID to the row
                row = {'HouseID': HouseID}
                row.update(house_data)
                writer.writerow(row)

    def WriteData(self):

        while True:

            addData = input("yes or no")

            if addData == 'yes':

                print("Write down the housing data you have found:\n")

                housingPrice = input("Price: ")
                housingType = input("Type: ")
                housingSquareMeters = input("Square Meters: ")
                housingBedrooms = input("Bedrooms: ")
                housingBathrooms = input("Bathrooms: ")
                housingBuildYear = input("Build Year: ")
                housingUseableArea = input("Useable Area: ")
                housingSchoolDistance = input("School Distance: ")
                housingCondition = input("Condition: ")

                # Use the next available ID for HouseID
                houseID = len(self.data) + 1

                self.saveData(houseID, housingPrice, housingType, housingSquareMeters, housingBedrooms, housingBathrooms,
                              housingBuildYear, housingUseableArea, housingSchoolDistance, housingCondition)

            elif addData == 'no':
                exit()


# Example usage
housing = HousingPrices()
housing.WriteData()  # Input data via the console
housing.storeData()  # Save it to CSV

