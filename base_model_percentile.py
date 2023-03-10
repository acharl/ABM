import mesa
import random
import matplotlib.pyplot as plt
import statistics
import numpy as np

class MarketPlace(mesa.Agent):

    def __init__(self, unique_id, model):
        self.avg_reward = 0
        self.total_rewards = 0
        self.total_jobs_matched = 0
        self.total_jobs_registered = 0
        self.total_jobs_fulfilled = 0
        self.total_jobs_failed = 0 
        self.ninetieth_percentile = 0 # the ninetieth percentile with respect to processors' reputations
        self.rewards = [] 
        self.jobs_to_be_matched = {} # indexed by consumer 
        self.advertisements = {} # indexed by processor
        super().__init__(unique_id, model)
        
    def register_job(self, job):
        self.total_jobs_registered += 1
        if (not self.match_job(job)):
            self.jobs_to_be_matched[job['consumer_id']] = job 
        
    def register_advertisement(self, advertisement):
        self.advertisements[advertisement["processor_id"]] = advertisement

    def match_job(self, job): 
        consumer = self.model.consumers[job["consumer_id"]]
        matches = []
        def is_within_budget(advertisement):
            if (job["reward"] > advertisement["fee_per_job"] ): 
                return True

        def has_min_reputation(advertisement):
            matched_processor = self.model.processors[advertisement["processor_id"]]
            return matched_processor.get_reputation() >= job['min_reputation']
            # return matched_processor.get_reputation() >= self.ninetieth_percentile
        
        ads_with_capacity = [ad for ad in list(self.advertisements.values())] 
        ads_within_budget = [ad for ad in ads_with_capacity if is_within_budget(ad)]

        if job['min_reputation']:
            matches = [ad for ad in ads_within_budget if has_min_reputation(ad)] 
        else: 
            matches = ads_within_budget
            
        if len(matches) > 0: 
            random.shuffle(matches)
            slots = job['slots']
            candidates = [self.model.processors[m["processor_id"]] for m in matches][:slots]

            if (len(candidates) == slots):
                for candidate in candidates: 
                    candidate.process_job(job, self.avg_reward, slots)
                self.jobs_to_be_matched[job['consumer_id']] = None
                consumer.on_process_job(job)
                self.total_jobs_matched += 1
                self.total_rewards += job['reward']
                self.avg_reward = self.total_rewards / self.total_jobs_matched 
                if (random.uniform(0, 1) > 0.9): 
                    self.ninetieth_percentile = np.percentile([p.reputation for p in self.model.processors.values()], 90)
                return True
        
            else: 
                return False            
        else: 
            return False
        
    def step(self):
        for job in [self.jobs_to_be_matched[i] for i in self.jobs_to_be_matched if self.jobs_to_be_matched[i] is not None]:
            self.match_job(job)

class ConsumerAgent(mesa.Agent):
    
    job = {}
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.max_fee_per_job = random.randint(75, 125)
        self.reward = 0.5 * self.max_fee_per_job
        self.slots = 1 #random.randint(1, 10)
        self.has_multiple_slots = random.uniform(0, 1) > 0.5 
        self.has_matched_job = False
        self.expects_min_reputation = True if random.uniform(0, 1) < 0.5 else None
        self.min_reputation = random.uniform(0.8, 1)
        self.allocated_jobs = 0

    def on_process_job(self, job):
        self.allocated_jobs += 1
        self.reward = job['reward']
        self.has_matched_job = True

    def step(self):
        if (self.reward * 1.1 <= self.max_fee_per_job):
            self.reward *= 1.1
        job = { 
            "reward": self.reward * self.slots if self.has_multiple_slots else self.reward, 
            # "min_reputation": None,
            # "min_reputation": self.expects_min_reputation,
            "min_reputation": self.min_reputation if random.uniform(0, 1) < 0.5 else None,
            "slots": self.slots if self.has_multiple_slots else 1, 
            "consumer_id": self.unique_id
        }
        self.model.market_place.register_job(job) 
           
            
class TransmitterAgent(mesa.Agent):
    weights = []
    has_open_advertisement = False
    income = 0 
    
    advertisement = {}
    r = 0 
    s = 0 
    

    # type ??? [low, medium, high] depending on success_rate
    def __init__(self, success_rate, type, unique_id, model):
        super().__init__(unique_id, model)
        self.min_fee_per_job = random.randint(50, 100)
        self.fee_per_job = 80
        self.success_rate = success_rate
        self.type = type
        self.reputation = self.calculate_reputation()

    def update_reputation(self, job, avg_reward, is_fulfilled):
        w = job['reward'] / (job['reward'] + avg_reward)
        ?? = self.model.lmbda
        self.r *= ?? 
        self.s *= ?? 
        if is_fulfilled:      
            self.model.market_place.total_jobs_fulfilled += 1
            self.r += w
        else: 
            self.s += w   
            self.model.market_place.total_jobs_failed += 1
        self.reputation = self.calculate_reputation()

    def calculate_reputation(self):
        return (self.r+1) / (self.r+self.s+2)
         
    def get_reputation(self): 
        ?? = self.model.lmbda
        mu = ((1/(1-??)) + 1)/(((1/(1-??)) + 2))
        rep = self.reputation
        return rep / mu

    def get_weights(self): 
        return sum(self.weights)/len(self.weights)

    def process_job(self, job, avg_reward, slots = None):
        self.has_open_advertisement = False
        self.model.market_place.rewards.append(job['reward'])
        is_fulfilled = random.uniform(0, 1) < self.success_rate
        if is_fulfilled: 
            if slots > 1: 
                self.income += job['reward']/slots
            else: 
                self.income += job['reward']
            self.update_reputation(job, avg_reward, True)
        else: 
            self.income -= job['reward'] 
            self.update_reputation(job, avg_reward, False) 
        self.register()


    def get_income(self): 
        return self.income


    def register(self): 
        self.advertisement = {
            "fee_per_job": self.fee_per_job, 
            "processor_id": self.unique_id
        }
        self.model.market_place.register_advertisement(self.advertisement)
            

    def step(self):
        self.register()
        pass

def compute_gini(processor_wealths): 
    x = sorted(processor_wealths)
    N = len(processor_wealths)
    B = sum(xi * (N - i) for i, xi in enumerate(x)) / (N * sum(x))
    return 1 + (1 / N) - 2 * B

class ReputationModel(mesa.Model):
    success_rates = [0.8, 0.9, 0.99]

    types = ['low', 'medium', 'high', 'ultra']

    def __init__(self, M, N):

        self.num_processors = M
        self.num_consumers = N

        self.lmbda = 0.98

        self.schedule = mesa.time.BaseScheduler(self)

        self.consumers = {}
        self.processors = {}

        # introduce three kinds of processors with different success_rates 
        for i in range(len(self.success_rates)): 
            for j in range(int(M/len(self.success_rates))): 
                 unique_id = "P_" + self.types[i] + '_' + str(j)
                 processor_agent = TransmitterAgent(self.success_rates[i], self.types[i], unique_id, self)
                 self.schedule.add(processor_agent)
                 self.processors[unique_id] = processor_agent

        # for j in range(30):
        #      unique_id = "P_" + self.types[3] + '_' + str(j)
        #      processor_agent = TransmitterAgent(0.999, self.types[3], unique_id, self)
        #      self.schedule.add(processor_agent)
        #      self.processors[unique_id] = processor_agent        

        for i in range(self.num_consumers):
            unique_id = "C_"+str(i)
            consumer_agent = ConsumerAgent("C_"+str(i), self)
            self.consumers[unique_id] = consumer_agent
            self.schedule.add(consumer_agent)

        self.market_place = MarketPlace(1, self)
        self.schedule.add(self.market_place)


    def step(self):
        self.schedule.step()

##################################################################################################

incomes_by_type = {}
reputations_by_type = {}
weights_by_type = {}

types = ['low', 'medium', 'high']

all_incomes = []
failure_rates = []
avg_reward = []
total_rewards = []
total_jobs_matched = []
total_jobs_registered = []
for i in range(1): 
    model = ReputationModel(300,900) 
    for j in range(200):
        model.step()

    for type in model.types: 
        processors_by_type = [p if p.type == type else None for p in model.processors.values()]
        incomes = [p.get_income() for p in processors_by_type if p is not None]
        all_incomes+=incomes
        reputations = [p.get_reputation() for p in processors_by_type if p is not None]

        if type in incomes_by_type: 
            incomes_by_type[type] += (incomes)
        else: 
            incomes_by_type[type] = incomes
        if type in reputations_by_type: 
            reputations_by_type[type] += (reputations)
        else: 
            reputations_by_type[type] = reputations

    avg_prices = sum(model.market_place.rewards)/model.market_place.total_jobs_matched

    avg_reward.append(avg_prices)
    failure_rate = model.market_place.total_jobs_failed/(model.market_place.total_jobs_failed + model.market_place.total_jobs_fulfilled)
    failure_rates.append(failure_rate)
    total_rewards.append(model.market_place.total_rewards)
    total_jobs_matched.append(model.market_place.total_jobs_matched)
    total_jobs_registered.append(model.market_place.total_jobs_registered)

colors = ['blue','red','green', 'purple']


fig = plt.figure(figsize = (7, 5))

for i, type in enumerate(types): 
    incomes = incomes_by_type[type]
    reputations = reputations_by_type[type]



    print('############### ' + type + ' ###############')
    print('MAX INC ' + str(max(incomes)))
    print('MIN INC ' + str(min(incomes)))
    print('AVG INC ' + str(sum(incomes)/len(incomes)))
    print('STDev INC ' + str(statistics.stdev(incomes)))

    print('--------------------------')
    print('MAX REP ' + str(max(reputations)))
    print('MIN REP ' + str(min(reputations)))
    print('AVG REP ' + str(sum(reputations)/len(reputations)))
    print('STDev REP ' + str(statistics.stdev(reputations)))
    print('\n')



    plt.scatter(incomes, reputations, c = colors[i], s=1)

plt.xlabel('Income')
plt.ylabel('Reputation')

print('\n')
print('AVG failure rate', str(sum(failure_rates)/len(failure_rates)))
print('\n')
print('AVG PRICES', str(sum(avg_reward)/len(avg_reward)))
print('\n')
print('AVG total_rewards', str(sum(total_rewards)/len(total_rewards)))
print('\n')
print('AVG total_jobs_matched', str(sum(total_jobs_matched)/len(total_jobs_matched)))
print('\n')
print('AVG total_jobs_registered', str(sum(total_jobs_registered)/len(total_jobs_registered)))
print('\n')
print('total_jobs_matched', total_jobs_matched)
print('\n')


plt.savefig('PLOT.pdf')

print('##############################')
gini = compute_gini(all_incomes)
print('GINI ' + str(gini))
print('##############################')
